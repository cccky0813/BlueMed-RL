"""③ Answer Agents 之二：Answer Reviewer（输出层 guardrail）。

两层审核，先便宜后昂贵：
1. 规则检查（确定性、免费）：引用编号必须真实存在于证据中；必须有引用；诊断 / 用药类问题必须有安全提示
2. 大模型检查（语义）：回答中有没有证据不支持的内容（幻觉）；证据够不够回答问题
结论：pass → 结束；rewrite → 打回 Generator；need_evidence → 打回 Expert 补查。最多打回 MAX_REVIEW_ROUNDS 次。
"""

import re

from langchain_core.messages import HumanMessage
from langgraph.graph import END

from bluemed.agents.answer_generator import format_evidence
from bluemed.agents.expert_agent import MAX_TOOL_CALLS
from bluemed.llm import structured_invoke
from bluemed.schemas import Answer, Evidence, Review
from bluemed.state import AgentState

MAX_REVIEW_ROUNDS = 2
EXTRA_TOOL_CALLS = 3                     # 打回 Expert 补查时，额外给的工具调用次数
DIAGNOSIS_WORDS = ["什么病", "是不是", "可能是", "得了什么", "怎么回事", "严重吗"]
DRUG_WORDS = ["药", "用药", "吃什么药", "服用"]

REVIEWER_PROMPT = """你是医学回答审核员（Answer Reviewer）。检查下面的回答是否可以发给用户。

逐条检查：
1. 忠实性：回答中的每个事实是否都能在【证据】中找到依据？出现证据里没有的事实（哪怕是常识）就是不合格，verdict = rewrite，reason 指出具体是哪句话。
2. 充分性：只有当问题的【核心诉求】完全没有任何证据覆盖，而且能明确说出换哪个关系或实体可能查到时
   （例如问"并发症的检查"，证据里却只有并发症、没有任何检查），才判 need_evidence，missing 写清还要查什么。
   证据数量少、不够全面，不算证据不足：只要回答如实说明了证据范围（如"图谱只记录了这几项"），就应 pass。
   回答已如实说明"图谱中没有相关信息"时，也应 pass。
3. 以上都没问题 → verdict = pass。"""


def rule_check(question: str, answer: Answer, evidence: dict[str, Evidence]) -> Review | None:
    """确定性规则检查。发现问题返回 Review(rewrite)，通过返回 None。"""
    inline_ids = set(re.findall(r"E[0-9a-f]{6}", answer.answer))
    fake = (set(answer.citations) | inline_ids) - set(evidence)
    if fake:
        return Review(verdict="rewrite", reason=f"引用了不存在的证据编号 {sorted(fake)}，只能引用【证据】中列出的编号。")
    if not answer.insufficient and not (answer.citations or inline_ids):
        return Review(verdict="rewrite", reason="回答没有引用任何证据编号，每个事实都必须标注依据。")
    needs_notice = any(w in question for w in DIAGNOSIS_WORDS + DRUG_WORDS)
    if needs_notice and not answer.safety_notice.strip():
        return Review(verdict="rewrite", reason="问题涉及疾病判断或用药，safety_notice 不能为空。")
    return None


def reviewer_node(state: AgentState) -> dict:
    answer, evidence = state["answer"], state.get("evidence", {})
    review = rule_check(state["question"], answer, evidence)
    if review is None:
        prompt = (f"{REVIEWER_PROMPT}\n\n【证据】\n{format_evidence(evidence)}\n\n【用户问题】\n{state['question']}"
                  f"\n\n【待审核回答】\n{answer.answer}\n安全提示：{answer.safety_notice}\n标记证据不足：{answer.insufficient}")
        review = structured_invoke(Review, prompt)

    rounds = state.get("review_rounds", 0)
    update: dict = {"review": review, "next_step": "end"}
    if review.verdict != "pass" and rounds < MAX_REVIEW_ROUNDS:
        update["review_rounds"] = rounds + 1
        if review.verdict == "rewrite":
            update["next_step"] = "generator"
        else:
            # 把审核意见作为一条新消息交给 Expert，并额外给几次工具调用的额度
            update["next_step"] = "expert"
            update["messages"] = [HumanMessage(f"【Reviewer 反馈】证据不足：{review.missing or review.reason}。请补充检索。")]
            update["tool_calls"] = min(state.get("tool_calls", 0), MAX_TOOL_CALLS - EXTRA_TOOL_CALLS)
    return update


def route_after_review(state: AgentState) -> str:
    """路由只读状态：去向已经由 reviewer_node 决定好写在 next_step 里。"""
    return {"generator": "generator", "expert": "expert"}.get(state.get("next_step"), END)
