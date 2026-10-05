"""演示 Reviewer 如何拦截幻觉：故意构造两个有问题的回答交给 reviewer_node。
1. 编造证据编号 → 规则层拦截（不调用大模型）
2. 引用的编号是真的，但内容超出证据（"海鲜、啤酒"是常识，不是图谱证据）→ 大模型层拦截
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bluemed.agents.answer_reviewer import reviewer_node
from bluemed.schemas import Answer
from bluemed.tools.kg_tools import kg_search

_, evidence = kg_search.func("痛风", "SHOULD_NOT_EAT")
ev = {e.id: e for e in evidence}
real_id = evidence[0].id
question = "我爸有痛风，平时饮食上有什么忌口？"

cases = {
    "编造编号": Answer(answer="痛风患者应避免鸡肝 [Edeadbe]", citations=["Edeadbe"], safety_notice="", insufficient=False),
    "内容超出证据": Answer(answer=f"痛风患者应避免海鲜、啤酒和动物内脏 [{real_id}]", citations=[real_id],
                       safety_notice="", insufficient=False),
}
for name, answer in cases.items():
    out = reviewer_node({"question": question, "answer": answer, "evidence": ev})
    print(f"【{name}】{answer.answer}\n  → {out['review'].verdict}：{out['review'].reason}\n  → 下一步：{out['next_step']}\n")
