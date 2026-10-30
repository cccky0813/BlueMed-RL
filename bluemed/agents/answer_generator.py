"""③ Answer Agents 之一：Answer Generator。只根据 Expert 收集的证据生成结构化回答（带引用编号）。"""

from bluemed.llm import structured_invoke
from bluemed.memory.profile import describe
from bluemed.schemas import Answer, Evidence
from bluemed.state import AgentState

MAX_EVIDENCE_IN_PROMPT = 40

GENERATOR_PROMPT = """你是医学问答助手（Answer Generator）。根据下面的【证据】回答用户问题。

规则：
1. 只能使用证据中的信息，不要用自己的医学知识补充任何事实。
2. 每个事实性陈述后面标注它依据的证据编号，如 [E3c9314]；citations 列出所有引用过的编号。
3. 证据不足以回答问题时，insufficient 设为 true，并如实说明图谱里能查到什么、查不到什么。
4. 证据之间有矛盾时（例如同一种食物既宜吃又忌吃），指出矛盾，不要只挑一边说。
5. 涉及可能的疾病判断时，safety_notice 提醒"不能替代医生诊断，请及时就医"；涉及药物时提醒"请在医生指导下用药"。
6. 回答简洁、分点，面向普通用户。
7. answer 和 safety_notice 使用【回答语言】。证据是中文的：用英文回答时，把疾病、食物、药物等名词译成英文
   （可在括号里保留中文，如 chicken liver (鸡肝)），证据编号原样保留。"""

LANGUAGE_NAMES = {"zh": "中文", "en": "English"}


def format_evidence(evidence: dict[str, Evidence]) -> str:
    items = sorted(evidence.values(), key=lambda e: (e.hop, -e.score))[:MAX_EVIDENCE_IN_PROMPT]
    return "\n".join(e.to_text() for e in items) or "（没有检索到任何证据）"


def generator_node(state: AgentState) -> dict:
    prompt = (f"{GENERATOR_PROMPT}\n\n【证据】\n{format_evidence(state.get('evidence', {}))}"
              f"\n\n【用户档案】（回答时考虑，例如涉及过敏的药物要提醒）\n{describe(state.get('profile') or {})}"
              f"\n\n【用户问题】\n{state['question']}（用户原话：{state.get('user_input', '')}）"
              f"\n\n【回答语言】\n{LANGUAGE_NAMES.get(state.get('language', 'zh'), '中文')}")
    review = state.get("review")
    if review and review.verdict == "rewrite":                       # 被 Reviewer 打回：带上审核意见重写
        prompt += f"\n\n【上一版回答】\n{state['answer'].answer}\n\n【审核意见，必须针对性修改】\n{review.reason}"
    answer = structured_invoke(Answer, prompt)
    return {"answer": answer}
