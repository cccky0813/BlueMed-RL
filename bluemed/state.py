"""所有 Agent 共享的状态（LangGraph 的 State）。后续阶段会继续往里加字段（解析结果、回答、审核意见、记忆等）。"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from bluemed.schemas import Answer, Evidence, Review


def merge_evidence(old: dict[str, Evidence], new: dict[str, Evidence]) -> dict[str, Evidence]:
    """证据的 reducer：按编号合并，已有的不覆盖（编号由内容决定，相同编号就是同一条事实）。"""
    return {**new, **(old or {})}


class AgentState(TypedDict, total=False):
    question: str                                                  # 用户问题
    messages: Annotated[list[AnyMessage], add_messages]            # Expert 的工作记录：思考、工具调用、工具结果（追加）
    evidence: Annotated[dict[str, Evidence], merge_evidence]       # 所有工具查到的证据（按编号合并）
    tool_calls: int                                                # 已执行的工具调用次数（覆盖）
    answer: Answer                                                 # Answer Generator 的结构化回答
    review: Review                                                 # Answer Reviewer 的审核结论
    review_rounds: int                                             # 已打回的次数
    next_step: str                                                 # Reviewer 决定的去向：end / generator / expert
