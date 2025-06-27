"""State management for the multi-agent system."""
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.graph import END


class CodeMessageState(BaseModel):
    """记录Agent通信过程的状态"""
    task: str = Field(description="The task")
    messages: HumanMessage = Field(description="The Agent message")
    feedback: str = Field(description="The feedback")
    code: str = Field(description="The output code")
    executed_nodes: List = Field(description="The executed nodes")
    prompt_token: int
    completion_token: int
    next_node: Literal["plan_node", "research_node", "code_generate_node", "code_review_node", "test_node", END] # type: ignore


class MathMessageState(BaseModel):
    """记录Agent通信过程的状态"""
    task: str = Field(description="The task")
    messages: List = Field(description="The Agent message")
    answer: str = Field(description="The math answer")
    code: str = Field(description="The output code")
    executed_nodes: List = Field(description="The executed nodes")
    prompt_token: int
    completion_token: int
    next_node: Literal["MathSolverAgent", "MathematicalAnalystAgent", "ProgrammingExpertAgent", "InspectorAgent", END]   # type: ignore