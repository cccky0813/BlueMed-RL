"""运行完整系统（Expert → Answer Generator → Answer Reviewer），打印每个 Agent 的动作和最终回答。
用法：python scripts/run_pipeline.py "痛风能喝啤酒吗？"
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, ToolMessage

from bluemed.graph import build_graph


def run(question: str):
    app = build_graph()
    print(f"\n{'=' * 70}\n问题：{question}\n{'=' * 70}")
    start, final = time.perf_counter(), {"question": question}
    for step in app.stream({"question": question}, stream_mode="updates"):
        for node, update in step.items():
            final.update(update)
            if node == "expert":
                msg = update["messages"][-1]
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    print("🔬 Expert 调用  " + "；".join(f"{c['name']}({c['args']})" for c in msg.tool_calls))
                else:
                    print(f"🔬 Expert  {msg.content}")
            elif node == "tools":
                print(f"🔧 工具执行 {len(update['messages'])} 个，累计证据 {len(update.get('evidence', {}))} 条（本轮新增）")
            elif node == "generator":
                a = update["answer"]
                print(f"✍️  Generator 生成回答（引用 {len(a.citations)} 条，证据不足={a.insufficient}）")
            elif node == "reviewer":
                r = update["review"]
                print(f"🧐 Reviewer  {r.verdict}：{r.reason[:80]}  → 下一步 {update['next_step']}")
    a = final["answer"]
    print(f"\n【最终回答】\n{a.answer}\n【安全提示】{a.safety_notice or '（无）'}")
    print(f"—— 打回 {final.get('review_rounds', 0)} 次，耗时 {time.perf_counter() - start:.1f}s")
    return final


if __name__ == "__main__":
    for q in sys.argv[1:] or ["我爸有痛风，平时饮食上有什么忌口？", "痛风能喝啤酒吗？", "头痛还发烧，可能是什么病？吃什么药？"]:
        run(q)
