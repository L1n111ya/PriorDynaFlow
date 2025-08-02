"""State management for the multi-agent system."""
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.graph import END

_STATE_TYPES_REGISTRY = {}


def register_state_type(node_name: str):
    def decorator(cls):
        _STATE_TYPES_REGISTRY[node_name] = cls
        return cls

    return decorator


def get_state_type(node_name: str) -> type:
    return _STATE_TYPES_REGISTRY.get(node_name, CodeMessageState)  # 默认 fallback 到 CodeMessageState


class BaseMessageState(BaseModel):
    """"""
    task: str = Field(description="The task")
    messages: HumanMessage = Field(description="The Agent message")
    code: str = Field(description="The output code")
    executed_nodes: List = Field(description="The executed nodes")
    prompt_token: int = 0
    completion_token: int = 0
    next_node: str


@register_state_type("plan_node")
@register_state_type("research_node")
@register_state_type("code_generate_node")
@register_state_type("code_review_node")
@register_state_type("test_node")
@register_state_type(END)
class CodeMessageState(BaseMessageState):
    """记录Agent通信过程的状态"""
    feedback: str = Field(description="The feedback")
    next_node: Literal[
        "plan_node", "research_node", "code_generate_node", "code_review_node", "test_node", END]  # type: ignore


@register_state_type("MathSolverAgent")
@register_state_type("MathematicalAnalystAgent")
@register_state_type("ProgrammingExpertAgent")
@register_state_type("InspectorAgent")
class MathMessageState(BaseMessageState):
    """记录Agent通信过程的状态"""
    answer: str = Field(description="The math answer")
    next_node: Literal[
        "MathSolverAgent", "MathematicalAnalystAgent", "ProgrammingExpertAgent", "InspectorAgent", END]  # type: ignore


@register_state_type("AnalystAgent")
@register_state_type("InspectorAgent")
@register_state_type("PlanAgent")
@register_state_type("ProgrammingAgent")
@register_state_type("CodeAuditorAgent")
@register_state_type("TestEngineerAgent")
class HybridMessageState(BaseMessageState):
    """记录混合任务（数学+代码）的Agent通信过程状态"""
    answer: str = Field(description="The final answer")
    feedback: str = Field(description="The feedback")
    next_node: Literal[
        "AnalystAgent", "ProgrammingAgent", "InspectorAgent", "PlanAgent", "CodeAuditorAgent", "TestEngineerAgent", END]  # type: ignore