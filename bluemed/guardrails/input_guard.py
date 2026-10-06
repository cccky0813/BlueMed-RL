"""输入层 guardrail 的规则部分（确定性、零延迟）。语义层面的判断由 Parser Agent 的大模型完成。

为什么急症要有规则兜底：急症漏判代价极高，不能只依赖大模型（慢、偶尔误判）；规则宁可多报。
"""

import re

EMERGENCY_PATTERNS = [
    "胸痛", "胸口痛", "喘不上气", "呼吸困难", "昏迷", "意识不清", "叫不醒", "大出血", "吐血", "抽搐",
    "口眼歪斜", "半边身体", "说话不清", "自杀", "不想活", "割腕", "吞了", "误服", "过量服用", "休克",
]
INJECTION_PATTERNS = [
    r"忽略(之前|以上|前面|上面)?的?(所有)?(指令|规则|设定|要求)", r"ignore (all )?(previous|above|prior) instructions",
    r"(系统|system)\s*(提示词|prompt)", r"你现在(是|扮演)", r"developer mode", r"越狱", r"jailbreak",
]

EMERGENCY_REPLY = ("⚠️ 你描述的情况可能是急症，请立即拨打 120 或尽快前往最近的急诊，不要等待线上回复。"
                   "在等待救援时保持患者安静、不要随意喂食或喂药。")
OUT_OF_SCOPE_REPLY = "我是医疗健康问答助手，只能回答疾病、症状、用药、饮食、检查等健康相关的问题。"
INJECTION_REPLY = "抱歉，我不能执行这个请求。我只能基于医学知识图谱回答健康相关的问题。"


def rule_check(text: str) -> str | None:
    """返回 'emergency' / 'injection'，未命中返回 None。急症优先。"""
    if any(p in text for p in EMERGENCY_PATTERNS):
        return "emergency"
    if any(re.search(p, text, re.IGNORECASE) for p in INJECTION_PATTERNS):
        return "injection"
    return None
