"""所有 Agent 共享的状态（LangGraph 的 State）。后续阶段会继续往里加字段（解析结果、回答、审核意见、记忆等）。"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from bluemed.schemas import Answer, Evidence, Review


def merge_evidence(old: dict[str, Evidence], new: dict[str, Evidence]) -> dict[str, Evidence]:
    """证据的 reducer：按编号合并，已有的不覆盖（编号由内容决定，相同编号就是同一条事实）。
    传入 None 表示清空——每一轮新问题开始时，由 Parser 清空上一轮的证据（工作记忆不能跨轮）。"""
    if new is None:
        return {}
    return {**new, **(old or {})}


class AgentState(TypedDict, total=False):
    # ---- 本轮输入（Parser 写入）
    user_input: str                                                # 用户原话
    image_id: str | None                                           # 本轮上传的图片编号（没有为 None；每轮由调用方传入）
    question: str                                                  # Parser 改写后、能独立理解的问题（后续 Agent 都用它）
    parsed: dict                                                   # LoRA 解析模型抽取的实体 / 关系（给 Expert 的线索）
    category: str                                                  # medical / emergency / out_of_scope / injection
    language: str                                                  # 用户提问的语言 zh / en：内部一律用中文检索，回答用用户的语言
    # ---- 记忆
    conversation: Annotated[list[dict], operator.add]              # 短期记忆：本会话的问答历史（追加，随 checkpointer 持久化）
    profile: dict                                                  # 长期记忆：用户健康档案（从 Store 读出，供本轮使用）
    # ---- 本轮工作记忆（每轮开始时清空）
    messages: Annotated[list[AnyMessage], add_messages]            # Expert 的工作记录：思考、工具调用、工具结果（追加）
    evidence: Annotated[dict[str, Evidence], merge_evidence]       # 所有工具查到的证据（按编号合并）
    tool_calls: int                                                # 已执行的工具调用次数（覆盖）
    answer: Answer                                                 # Answer Generator 的结构化回答
    review: Review                                                 # Answer Reviewer 的审核结论
    review_rounds: int                                             # 已打回的次数
    next_step: str                                                 # Reviewer 决定的去向：end / generator / expert
