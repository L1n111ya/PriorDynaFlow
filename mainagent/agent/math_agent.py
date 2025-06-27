"""Node implementations for the multi-agent system."""
import random
import re
from typing import Annotated, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from mainagent.agent.base_agent import BaseNode, logger
from mainagent.prompt.math_prompt import MATH_SOLVER_PROMPT, MATHEMATICAL_ANALYST_PROMPT, PROGRAMMING_EXPERT_PROMPT, \
    INSPECTOR_PROMPT, FEW_SHOT_DATA
from mainagent.state import MathMessageState
from mainagent.tools.utils import python_repl_tool
from mainagent.tools.coding.python_executor import execute_code_get_return, PyExecutor
from mainagent.tools.math.get_predict import get_predict


class MathSolverAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.sys_prompt = system_prompt + MATH_SOLVER_PROMPT
        """+ FEW_SHOT_DATA["Math Solver"]"""

    def _execute_node(self, state: MathMessageState, next_avail_nodes: List) -> MathMessageState:
        """Math Solver Agent"""
        logger.info("=====MATH SOLVER AGENT RUN=====")
        message = HumanMessage(content=state.task)

        result = self.llm.invoke(
            self.get_prompt(self.sys_prompt, next_avail_nodes, message, state.executed_nodes)
        )

        print(f"MathSolverAgent: {result.content}")

        # pattern = r'\$\$(.*?)\$\$'
        # match = re.search(pattern, result.content, re.DOTALL)
        # answer = match.group(1).strip() if match else ""
        answer = get_predict(result.content)

        goto = self.get_next_node(result)

        if answer is None and goto == END:
            goto = random.choice(next_avail_nodes)
            answer = ""
        elif answer is None:
            answer = ""

        input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
        output_tokens = result.usage_metadata["output_tokens"] + state.completion_token

        messages = HumanMessage(
            content=result.content,
            name="math_solver"
        )

        # Update executed nodes list
        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return MathMessageState(
            task=state.task,
            messages=messages,
            code=state.code,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            answer=answer,
            next_node=goto
        )


class MathematicalAnalystAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.tools = python_repl_tool
        self.sys_prompt = system_prompt + MATHEMATICAL_ANALYST_PROMPT
        """+ FEW_SHOT_DATA["Mathematical Analyst"]"""
        # self.math_analyst_agent = create_react_agent(
        #     llm,
        #     tools=[self.tools],
        #     prompt=self.sys_prompt
        # )

    def _execute_node(self, state: MathMessageState, next_avail_nodes: List) -> MathMessageState:
        """Mathematical Analyst Agent"""
        logger.info("=====MATHEMATICAL ANALYST AGENT RUN=====")
        # is_solved, feedback, state = PyExecutor().execute(state.code, self.internal_tests, timeout=10)
        solve = execute_code_get_return(state.code)
        message = HumanMessage(
            content=f"""
                    用户任务: {state.task},
                    Python代码: {state.code},
                    代码运行结果: {solve}
                    上一节点信消息: {state.messages}
                    """
        )

        result = self.llm.invoke(
            self.get_prompt(self.sys_prompt, next_avail_nodes, message, state.executed_nodes)
        )
        print(f"MathematicalAnalystAgent: {result.content}")

        # pattern = r'\$\$(.*?)\$\$'
        # match = re.search(pattern, result.content, re.DOTALL)
        # if not match:
        #     logger.info("No answer found in the result.")
        #     # raise
        # answer = match.group(1).strip() if match else state.answer
        answer = get_predict(result.content)

        input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
        output_tokens = result.usage_metadata["output_tokens"] + state.completion_token

        messages = HumanMessage(
            content=result.content,
            name="mathematical_analyst"
        )

        goto = self.get_next_node(result)

        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return MathMessageState(
            task=state.task,
            messages=messages,
            code=state.code,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            answer=answer if answer else state.answer,
            next_node=goto
        )


class ProgrammingExpertAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.sys_prompt = system_prompt + PROGRAMMING_EXPERT_PROMPT
        """+ FEW_SHOT_DATA["Programming Expert"]"""

    def _execute_node(self, state: MathMessageState, next_avail_nodes: List) -> MathMessageState:
        """Programming Expert Agent"""
        logger.info("=====PROGRAMMING EXPERT AGENT RUN=====")
        message = HumanMessage(
            content=f"""
                    用户任务: {state.task},
                    上一节点信消息: {state.messages}
                    """
        )

        result = self.llm.invoke(
            self.get_prompt(self.sys_prompt, next_avail_nodes, message, state.executed_nodes)
        )
        print(f"ProgrammingExpertAgent: {result.content}")

        # 过滤掉注释并提取剩余代码
        code_pattern = r'```python\n(.*?)```'
        code_match = re.search(code_pattern, result.content, re.DOTALL)
        clean_code = code_match.group(1).strip() if code_match else None

        # answer = None
        if clean_code:
            answer = execute_code_get_return(clean_code)
        else:
            answer = get_predict(result.content)

        input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
        output_tokens = result.usage_metadata["output_tokens"] + state.completion_token

        goto = self.get_next_node(result)

        messages = HumanMessage(
            content=result.content,
            name="programming_expert"
        )

        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return MathMessageState(
            task=state.task,
            messages=messages,
            code=clean_code,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            answer=str(answer) if answer else state.answer,
            next_node=goto
        )


class InspectorAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.tools = python_repl_tool
        self.sys_prompt = system_prompt + INSPECTOR_PROMPT
        # self.inspector_agent = create_react_agent(
        #     llm,
        #     tools=[self.tools],
        #     prompt=self.sys_prompt
        # )

    def _execute_node(self, state: MathMessageState, next_avail_nodes: List) -> MathMessageState:
        """Inspector Agent"""
        logger.info("=====INSPECTOR AGENT RUN=====")
        message = HumanMessage(
            content=f"""
                    用户任务: {state.task},
                    Python代码: {state.code},
                    当前答案: {state.answer},
                    上一节点信消息: {state.messages}
                    """
        )

        result = self.llm.invoke(
            self.get_prompt(self.sys_prompt, next_avail_nodes, message, state.executed_nodes)
        )
        print(f"InspectorAgent: {result.content}")

        # pattern = r'\$\$(.*?)\$\$'
        # match = re.search(pattern, result.content, re.DOTALL)
        # answer = match.group(1).strip() if match else state.answer
        # 过滤掉注释并提取剩余代码
        code_pattern = r'```python\n(.*?)```'
        code_match = re.search(code_pattern, result.content, re.DOTALL)
        clean_code = code_match.group(1).strip() if code_match else None

        answer = None
        if clean_code:
            answer = execute_code_get_return(clean_code)
        if get_predict(result.content):
            answer = get_predict(result.content)

        input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
        output_tokens = result.usage_metadata["output_tokens"] + state.completion_token

        goto = self.get_next_node(result)

        messages = HumanMessage(
            content=result.content,
            name="inspector"
        )

        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return MathMessageState(
            task=state.task,
            messages=messages,
            code=state.code,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            answer=str(answer) if answer else state.answer,
            next_node=goto
        )


