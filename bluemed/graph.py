"""把三个 Agent 组装成完整的 LangGraph 图。

  START → parser ─拦截（急症/越界/注入）────────────────────────────────▶ remember → END
            └─医疗问题─▶ expert ⇄ tools → generator → reviewer ─pass──▶ remember
                           ▲                  ▲           ├─rewrite─────▶ generator
                           └──────────────────┼───────────┴─need_evidence▶ expert

记忆：
  短期记忆：checkpointer（每个会话一个 thread_id，State 自动持久化，conversation 字段保存问答历史）
  长期记忆：store（按 user_id 保存健康档案，跨会话共享）
  不传时用内存版（测试 / 脚本）；Web 服务传入 SQLite 版（bluemed/memory/persistence.py），重启不丢
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from bluemed.agents.answer_generator import generator_node
from bluemed.agents.answer_reviewer import reviewer_node, route_after_review
from bluemed.agents.expert_agent import expert_node, route_after_expert, tools_node
from bluemed.agents.parser_agent import parser_node, remember_node, route_after_parser
from bluemed.memory.persistence import SERDE
from bluemed.state import AgentState


def build_graph(checkpointer=None, store=None):
    graph = StateGraph(AgentState)
    graph.add_node("parser", parser_node)
    graph.add_node("expert", expert_node)
    graph.add_node("tools", tools_node)
    graph.add_node("generator", generator_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("remember", remember_node)

    graph.add_edge(START, "parser")
    graph.add_conditional_edges("parser", route_after_parser, ["expert", "remember"])
    # Expert 的路由在"证据收集完"时返回 END；映射表把它接到 generator，Expert 模块本身不用改
    graph.add_conditional_edges("expert", route_after_expert, {"tools": "tools", END: "generator"})
    graph.add_edge("tools", "expert")
    graph.add_edge("generator", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_review, {"generator": "generator", "expert": "expert", END: "remember"})
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver(serde=SERDE), store=store or InMemoryStore())
