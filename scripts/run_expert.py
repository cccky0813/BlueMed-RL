"""运行 Expert Agent，打印完整的推理轨迹（每一轮的思考、工具调用、工具结果）。
用法：python scripts/run_expert.py "糖尿病的并发症需要做什么检查？"
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, ToolMessage

from bluemed.agents.expert_agent import build_expert_graph


def run(question: str):
    app = build_expert_graph()
    print(f"\n{'=' * 70}\n问题：{question}\n{'=' * 70}")
    start, final = time.perf_counter(), {}
    for state in app.stream({"question": question}, stream_mode="values"):
        final = state
        last = state["messages"][-1] if state.get("messages") else None
        if isinstance(last, AIMessage):
            if last.tool_calls:
                for c in last.tool_calls:
                    print(f"🤔 决定调用  {c['name']}({c['args']})")
            else:
                print(f"✅ 结束：{last.content}")
        elif isinstance(last, ToolMessage):
            batch = []                                     # 同一轮可能并行调用了多个工具，把这一轮的结果都打印出来
            for m in reversed(state["messages"]):
                if not isinstance(m, ToolMessage):
                    break
                batch.append(m)
            for m in reversed(batch):
                lines = m.content.splitlines()
                print(f"🔧 {m.name} 返回  {lines[0][:70]}{'  …共 %d 行' % len(lines) if len(lines) > 1 else ''}")
    print(f"—— 工具调用 {final.get('tool_calls', 0)} 次，收集证据 {len(final.get('evidence', {}))} 条，"
          f"耗时 {time.perf_counter() - start:.1f}s")
    return final


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "我爸有痛风，平时饮食上有什么忌口？",
        "糖尿病的并发症需要做什么检查？",
        "头痛还发烧，可能是什么病？",
        "糖尿病人是什么原因引起的，怎么预防？",
    ]
    for q in questions:
        run(q)
