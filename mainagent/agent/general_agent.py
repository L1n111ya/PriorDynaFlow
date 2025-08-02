import re
from typing import List

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END

from mainagent.agent.base_agent import logger
from mainagent.agent.base_agent import BaseNode
from mainagent.exceptions import NodeExecutionError
from mainagent.state import HybridMessageState
from mainagent.prompt.general_prompt import ANALYST_PROMPT, PROGRAMMING_PROMPT, \
    INSPECTOR_PROMPT, CODE_AUDITOR_PROMPT, PLAN_PROMPT, TEST_ENGINEER_PROMPT
from mainagent.tools.utils import python_repl_tool
from mainagent.tools.coding.python_executor import execute_code_get_return, PyExecutor
from mainagent.tools.math.get_predict import get_predict


class PlanAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.sys_prompt = system_prompt + PLAN_PROMPT
        
    def _execute_node(self, state: HybridMessageState, next_avail_nodes: List) -> HybridMessageState:
        """Hybrid Solver Agent"""
        print("=====PLAN AGENT RUN=====")
        message = HumanMessage(content=state.task)
        
        result = self.llm.invoke(
            self.get_prompt(self.sys_prompt, next_avail_nodes, message, state.executed_nodes)
        )

        answer = get_predict(result.content)

        # print(f"Plan Result: {result.content}")

        goto = self.get_next_node(result)

        input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
        output_tokens = result.usage_metadata["output_tokens"] + state.completion_token

        messages = HumanMessage(
            content=result.content,
            name="plan_node"
        )

        # Update executed nodes list
        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return HybridMessageState(
            task=state.task,
            messages=messages,
            code=state.code,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            answer=answer if answer else state.answer,
            feedback=state.feedback,
            next_node=goto
        )


class SolverAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.sys_prompt = system_prompt

    def _execute_node(self, state: HybridMessageState, next_avail_nodes: List) -> HybridMessageState:
        """Hybrid Solver Agent"""
        print("=====HYBRID SOLVER AGENT RUN=====")
        message = HumanMessage(content=state.messages)

        result = self.llm.invoke(
            self.get_prompt(self.sys_prompt, next_avail_nodes, message, state.executed_nodes)
        )

        answer = get_predict(result.content)

        goto = self.get_next_node(result)

        input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
        output_tokens = result.usage_metadata["output_tokens"] + state.completion_token

        messages = HumanMessage(
            content=result.content,
            name="solver_node"
        )

        # Update executed nodes list
        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return HybridMessageState(
            task=state.task,
            messages=messages,
            code=state.code,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            answer=answer if answer else state.answer,
            feedback=state.feedback,
            next_node=goto
        )


class AnalystAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.tools = python_repl_tool
        self.sys_prompt = system_prompt + ANALYST_PROMPT

    def _execute_node(self, state: HybridMessageState, next_avail_nodes: List) -> HybridMessageState:
        """Hybrid Analyst Agent"""
        
        print("=====ANALYST AGENT RUN=====")
        solve = execute_code_get_return(state.code)
        message = HumanMessage(
            content=f"""
                    用户任务: {state.task},
                    Python代码: {state.code},
                    代码运行结果: {solve}
                    上一节点消息: {state.messages}
                    """
        )

        result = self.llm.invoke(
            self.get_prompt(self.sys_prompt, next_avail_nodes, message, state.executed_nodes)
        )

        answer = get_predict(result.content)
        if "代码" in state.task:
            answer = state.code

        input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
        output_tokens = result.usage_metadata["output_tokens"] + state.completion_token

        messages = HumanMessage(
            content=result.content,
            name="hybrid_analyst"
        )

        goto = self.get_next_node(result)

        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return HybridMessageState(
            task=state.task,
            messages=messages,
            code=state.code,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            answer=answer if answer else state.answer,
            feedback=state.feedback,
            next_node=goto
        )


class ProgrammingAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.sys_prompt = system_prompt + PROGRAMMING_PROMPT

    def _execute_node(self, state: HybridMessageState, next_avail_nodes: List) -> HybridMessageState:
        """Programming Agent"""
        print("=====PROGRAMMING AGENT RUN=====")
        code = state.code
        messages = state.messages
        task = state.task

        enhanced_content = f"""
                        用户任务: {task},
                        其他节点消息: {messages.content},
                        当前代码: {code},
                        代码运行情况: {state.feedback}
                    """
        enhanced_messages = HumanMessage(
            content=enhanced_content,
            name="code_generator"
        )
        max_attempts = 3
        for attempt in range(max_attempts):
            result = self.llm.invoke(
                self.get_prompt(self.sys_prompt, next_avail_nodes, enhanced_messages, state.executed_nodes))

            code_pattern = r'```python\n(.*?)```'
            code_match = re.search(code_pattern, result.content, re.DOTALL)
            clean_code = code_match.group(1).strip() if code_match else ''

            if 'def' in clean_code and '\\u' not in clean_code:
                input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
                output_tokens = result.usage_metadata["output_tokens"] + state.completion_token
                new_messages = HumanMessage(
                    content=result.content,
                    name="code_generator"
                )
                code = clean_code
                match = re.search(r'(.*?)answer\s*=', code, re.DOTALL)
                if match and "代码" in task:
                    code = match.group(1)
                break
            else:
                enhanced_messages = HumanMessage(
                    content=enhanced_messages.content + "\n注意请生成包含在代码块中的代码",
                    name="inspector"
                )
        else:
            raise NodeExecutionError("Failed to generate valid code after multiple attempts.")

        print(f"Programming Result: {result.content}")
        print(f"Programming code: {code}")

        if clean_code:
            answer = execute_code_get_return(clean_code)
        else:
            answer = get_predict(result.content)

        if not answer:
            is_solved, feedback, _ = self.run_test(code)
            answer = code
            if is_solved:
                goto = END
            else:
                goto = self.get_next_node(result)
        else:
            answer = state.answer
            goto = self.get_next_node(result)

        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return HybridMessageState(
            task=state.task,
            messages=new_messages,
            code=code,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            answer=code if 'def' in answer else answer,
            feedback=state.feedback,
            next_node=goto
        )

class InspectorAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.tools = python_repl_tool
        self.sys_prompt = system_prompt + INSPECTOR_PROMPT

    def _execute_node(self, state: HybridMessageState, next_avail_nodes: List) -> HybridMessageState:
        """Inspector Agent"""
        print("=====INSPECTOR AGENT RUN=====")
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

        # 过滤掉注释并提取剩余代码
        code_pattern = r'```python\n(.*?)```'
        code_match = re.search(code_pattern, result.content, re.DOTALL)
        clean_code = code_match.group(1).strip() if code_match else ''

        answer = None
        if clean_code:
            answer = execute_code_get_return(clean_code)
        if get_predict(result.content):
            answer = get_predict(result.content)

        if "代码" in state.task:
            answer = clean_code
        else:
            answer = state.answer

        input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
        output_tokens = result.usage_metadata["output_tokens"] + state.completion_token

        goto = self.get_next_node(result)

        messages = HumanMessage(
            content=result.content,
            name="inspector"
        )

        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return HybridMessageState(
            task=state.task,
            messages=messages,
            code=clean_code,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            answer=str(answer) if answer else state.answer,
            feedback=state.feedback,
            next_node=goto
        )


class CodeAuditorAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.sys_prompt = system_prompt + CODE_AUDITOR_PROMPT

    def _execute_node(self, state: HybridMessageState, next_avail_nodes: List) -> HybridMessageState:
        """Code Auditor Agent"""
        print("=====CODE AUDITOR AGENT RUN=====")
        code = state.code
        task = state.task
        messages = HumanMessage(
            content=f"""
                        用户任务: {task},
                        其他节点消息: {state.messages.content},
                        当前代码: {code},
                        代码运行情况: {state.feedback}
                    """,
            name="code_reviewer"
        )

        result = self.llm.invoke(self.get_prompt(self.sys_prompt, next_avail_nodes, messages, state.executed_nodes))

        code_pattern = r'```python\n(.*?)```'
        code_match = re.search(code_pattern, result.content, re.DOTALL)
        clean_code = code_match.group(1).strip() if code_match else ''
        code = clean_code if 'def' in clean_code else code

        is_solved, feedback, _ = self.run_test(code)

        # print(f"Code Auditor Result: {result.content}")

        if is_solved:
            goto = END
        else:
            goto = self.get_next_node(result)
        input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
        output_tokens = result.usage_metadata["output_tokens"] + state.completion_token
        messages = HumanMessage(
            content=result.content,
            name="reviewer"
        )

        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return HybridMessageState(
            task=state.task,
            messages=messages,
            code=code if code else state.code,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            answer=code if 'def' in state.answer else state.answer,
            feedback=state.feedback,
            next_node=goto
        )


class TestEngineerAgent(BaseNode):
    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.sys_prompt = system_prompt + TEST_ENGINEER_PROMPT

    def _execute_node(self, state: HybridMessageState, next_avail_nodes: List) -> HybridMessageState:
        """Test Engineer Agent"""
        print("=====TEST ENGINEER AGENT RUN=====")
        code = state.code
        task = state.task
        messages = HumanMessage(
            content=f"""
                    用户任务: {task},
                    当前代码: {code},
                    代码运行情况: {state.feedback}
                    """
        )

        result = self.llm.invoke(
            self.get_prompt(self.sys_prompt, next_avail_nodes, messages, state.executed_nodes)
        )
        input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
        output_tokens = result.usage_metadata["output_tokens"] + state.completion_token

        messages = HumanMessage(
            content=result.content,
            name="tester"
        )

        # print(f"Test Engineer Result: {result.content}")

        goto = self.get_next_node(result) if result else END
        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return HybridMessageState(
            task=state.task,
            messages=messages,
            code=code,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            answer=code,
            feedback=state.feedback,
            next_node=goto
        )
