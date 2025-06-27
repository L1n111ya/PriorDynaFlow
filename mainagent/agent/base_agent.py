import logging
import re
from typing import List

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import END

from mainagent.state import CodeMessageState, MathMessageState
from mainagent.config import Config
from mainagent.exceptions import NodeExecutionError
from mainagent.tools.coding.python_executor import PyExecutor

logger = logging.getLogger(__name__)


class BaseNode:
    """Base class for all nodes in the system."""

    def __init__(self, name: str):
        self.name = name
        self.salaries = Config.NODE_SALARIES
        self.success_rate = 0.5
        self.trails = 0
        self.success = 0
        self.is_start = False
        self.test_cases = None

    @staticmethod
    def validate_state(state: MathMessageState) -> bool:
        """验证状态的有效性"""
        try:
            if not state:
                logger.error("State is None")
                return False
            if not state.messages:
                logger.error("Messages is empty")
                return False
            if not state.next_node:
                logger.error("Next node is not specified")
                return False
            return True
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            return False

    def node(self, state: MathMessageState, test_cases: List, next_avail_nodes: List):
        """处理节点逻辑"""
        if not self.validate_state(state):
            raise NodeExecutionError(f"Invalid state in {self.name}")
        try:
            # self.extract_example(state.task)
            self.test_cases = test_cases
            return self._execute_node(state, next_avail_nodes)
        except Exception as e:
            logger.error(f"Node execution failed: {e}", exc_info=True)
            raise NodeExecutionError(f"Failed to execute {self.name}: {str(e)}")

    def _execute_node(self, state: MathMessageState, next_avail_nodes: List):
        """实际的节点执行逻辑，由子类实现"""
        raise NotImplementedError("This method should be implemented by subclass")

    @staticmethod
    def get_prompt(sys_prompt: str, next_avail_nodes: List, messages: HumanMessage, executed_nodes: List,
                   is_agent: bool = False):
        """获取提示信息"""
        avail_nodes_details = "\n".join(f"{node}" for node in next_avail_nodes)
        prev_nodes = ",".join(executed_nodes)
        sys_prompt = sys_prompt.format(next_avail_nodes=avail_nodes_details, prev_nodes=prev_nodes)

        if is_agent:
            return {"messages": messages.content}
        return [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=messages.content)
        ]

    def update_success_rate(self, is_success: bool = False):
        """更新成功率"""
        self.trails += 1
        if is_success:
            self.success += 1
        self.success_rate = self.success / self.trails if self.trails > 0 else self.success_rate

    def get_next_node(self, last_message: BaseMessage) -> str:
        """根据最后一条消息内容决定下一个节点"""
        if not last_message or not last_message.content:
            return END

        comment_pattern = r'/\*.*?\*/'
        comment_match = re.search(comment_pattern, last_message.content, re.DOTALL)
        comment = comment_match.group(0) if comment_match else ''

        content = comment + ' ' + last_message.content
        if "FINAL ANSWER" in content.upper() or "END" in comment.upper():
            logger.info("=====END=====")
            return END

        # for node in ["research_node", "code_generate_node", "test_node", "code_review_node"]:
        #     if node in content.lower():
        #         return node
        # for agent in ["MathSolverAgent", "MathematicalAnalystAgent", "ProgrammingExpertAgent", "InspectorAgent"]:
        #     if agent.lower() in comment.lower():
        #         return agent
        for agent in self.salaries.keys():
            if agent.lower() in comment.lower():
                return agent

        logger.info("=====END=====")
        return END

    @staticmethod
    def get_search_tool():
        """Get DuckDuckGo search tool"""
        wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
        return DuckDuckGoSearchResults(api_wrapper=wrapper, source="text")

    def extract_example(self, prompt: str):
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        self.test_cases = results

    def run_test(self, code: str):
        is_solved, feedback, state = PyExecutor().execute(code, self.test_cases, timeout=10)
        return is_solved, feedback, state