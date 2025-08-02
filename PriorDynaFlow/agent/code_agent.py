"""Node implementations for the multi-agent system."""
import re
from typing import Annotated, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langchain_openai import ChatOpenAI


from PriorDynaFlow.agent.base_agent import BaseNode, logger
from PriorDynaFlow.prompt.code_prompt import PLAN_PROMPT, RESEARCH_PROMPT, CHECK_PROMPT, CODE_GENERATE_PROMPT, TEST_PROMPT
from PriorDynaFlow.state import CodeMessageState
from PriorDynaFlow.exceptions import NodeExecutionError
from PriorDynaFlow.tools.utils import python_repl_tool



class PlanNode(BaseNode):
    """Planning node that creates execution plans."""

    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.sys_prompt = system_prompt + PLAN_PROMPT

    def _execute_node(self, state: CodeMessageState, next_avail_nodes: List) -> CodeMessageState:
        """Execute planning logic"""
        logger.info("=====PLAN NODE RUN=====")

        try:
            # Get current state
            code = state.code
            messages = state.messages
            task = state.task

            # Call LLM
            result = self.llm.invoke(
                self.get_prompt(self.sys_prompt, next_avail_nodes, messages, state.executed_nodes)
            )

            # Get next node
            goto = self.get_next_node(result)

            # Update token counts
            input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
            output_tokens = result.usage_metadata["output_tokens"] + state.completion_token

            # Create new message
            messages = HumanMessage(
                content=result.content + state.messages.content,
                name="planner"
            )

            # Update executed nodes
            new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
            new_executed_nodes.append(goto)

            # Create new state
            return CodeMessageState(
                task=task,
                code=code,
                feedback=state.feedback,
                messages=messages,
                executed_nodes=new_executed_nodes,
                prompt_token=input_tokens,
                completion_token=output_tokens,
                next_node=goto
            )

        except Exception as e:
            logger.error(f"Plan node execution failed: {str(e)}")
            raise NodeExecutionError(f"Failed to execute plan node: {str(e)}")


class ResearchNode(BaseNode):
    """Research node that performs information gathering."""

    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        from langgraph.prebuilt import create_react_agent
        self.agent = create_react_agent(
            llm,
            tools=[self.get_search_tool()],
            prompt=""
        )
        self.sys_prompt = system_prompt + RESEARCH_PROMPT

    def _execute_node(self, state: CodeMessageState, next_avail_nodes: List) -> CodeMessageState:
        """Execute research logic"""
        logger.info("=====RESEARCH NODE RUN=====")

        try:
            code = state.code
            messages = state.messages
            task = state.task

            result = self.agent.invoke(
                self.get_prompt(self.sys_prompt, next_avail_nodes, messages, state.executed_nodes, True)
            )

            goto = self.get_next_node(result["messages"][-1])

            input_tokens = result["messages"][-1].usage_metadata["input_tokens"] + state.prompt_token
            output_tokens = result["messages"][-1].usage_metadata["output_tokens"] + state.completion_token

            messages = HumanMessage(
                content=result["messages"][-1].content,
                name="researcher"
            )

            new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
            new_executed_nodes.append(goto)

            return CodeMessageState(
                task=task,
                code=code,
                feedback=state.feedback,
                messages=messages,
                executed_nodes=new_executed_nodes,
                prompt_token=input_tokens,
                completion_token=output_tokens,
                next_node=goto
            )

        except Exception as e:
            logger.error(f"Research node execution failed: {str(e)}")
            raise NodeExecutionError(f"Failed to execute research node: {str(e)}")


class GeneratorNode(BaseNode):
    """代码生成节点"""

    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.sys_prompt = system_prompt + CODE_GENERATE_PROMPT

    def _execute_node(self, state: CodeMessageState, next_avail_nodes: List) -> CodeMessageState:
        logger.info("=====GENERATOR NODE RUN=====")
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
                # print(f"Generated code: {clean_code}")
                break
            else:
                enhanced_messages = HumanMessage(
                    content=enhanced_messages.content + "\n注意请生成包含在代码块中的代码",
                    name="inspector"
                )
        else:
            raise NodeExecutionError("Failed to generate valid code after multiple attempts.")

        is_solved, feedback, _ = self.run_test(code)

        if is_solved:
            goto = END
        else:
            goto = self.get_next_node(result)

        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return CodeMessageState(
            task=task,
            code=code,
            feedback=feedback,
            messages=new_messages,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            next_node=goto
        )


class CodeReviewNode(BaseNode):
    """代码检查节点"""

    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        self.llm = llm
        self.sys_prompt = system_prompt + CHECK_PROMPT

    def _execute_node(self, state: CodeMessageState, next_avail_nodes: List) -> CodeMessageState:
        logger.info("=====CODE REVIEW NODE RUN=====")
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

        # print(f"Reviewed code: {clean_code}")
        is_solved, feedback, _ = self.run_test(code)

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

        return CodeMessageState(
            task=task,
            code=code,
            messages=messages,
            feedback=feedback,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            next_node=goto
        )


class TestNode(BaseNode):
    """代码测试节点"""

    def __init__(self, llm: ChatOpenAI, system_prompt: str, name: str):
        super().__init__(name)
        # self.tools = python_repl_tool
        self.llm = llm
        # self.agent = create_react_agent(
        #     llm,
        #     tools=[self.tools] if self.tools else [],
        #     prompt=system_prompt + TEST_PROMPT
        # )
        self.sys_prompt = system_prompt + TEST_PROMPT

    def _execute_node(self, state: CodeMessageState, next_avail_nodes: List) -> CodeMessageState:
        logger.info("=====TEST NODE RUN=====")
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
        # print(f"test: {result.content}")
        input_tokens = result.usage_metadata["input_tokens"] + state.prompt_token
        output_tokens = result.usage_metadata["output_tokens"] + state.completion_token

        messages = HumanMessage(
            content=result.content,
            name="tester"
        )
        # if result is None:
        #     error_feedback = {
        #         "test_result": "failed",
        #         "reason": "测试节点多次执行失败，无法获取测试结果。",
        #         "code": code,
        #         "suggestion": "请检查代码逻辑或语法，重新生成代码。"
        #     }
        #     messages = HumanMessage(content=str(error_feedback), name="tester")
        # else:
        #     test_output = result["messages"][-1].content
        #     input_tokens = result["messages"][-1].usage_metadata["input_tokens"] + state.prompt_token
        #     output_tokens = result["messages"][-1].usage_metadata["output_tokens"] + state.completion_token
        #
        #     if "ERROR" in test_output or "FAILED" in test_output:
        #         error_type = ""
        #         suggestion = ""
        #         if "SyntaxError" in test_output:
        #             error_type = "SyntaxError"
        #             suggestion = "检查语法错误，比如括号是否闭合、冒号是否遗漏等。"
        #         elif "NameError" in test_output:
        #             error_type = "NameError"
        #             suggestion = "检查变量或函数名是否拼写错误或未定义。"
        #         elif "TypeError" in test_output:
        #             error_type = "TypeError"
        #             suggestion = "检查参数类型是否正确，函数调用是否传入了错误类型。"
        #         else:
        #             error_type = "OtherError"
        #             suggestion = "请根据错误信息修正代码。"
        #         error_feedback = {
        #             "test_result": "failed",
        #             "error_type": error_type,
        #             "error_message": test_output,
        #             "suggestion": suggestion
        #         }
        #         messages = HumanMessage(content=str(error_feedback), name="tester")
        #     else:
        #         messages = HumanMessage(content=test_output, name="tester")
        goto = self.get_next_node(result) if result else END
        new_executed_nodes = state.executed_nodes.copy() if state.executed_nodes else []
        new_executed_nodes.append(goto)

        return CodeMessageState(
            task=task,
            code=code,
            feedback=state.feedback,
            messages=messages,
            executed_nodes=new_executed_nodes,
            prompt_token=input_tokens,
            completion_token=output_tokens,
            next_node=goto
        )

