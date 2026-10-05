"""把各个 Agent 组装成完整的 LangGraph 图。

    START → expert ⇄ tools → generator → reviewer ─pass──────────▶ END
                ▲                ▲            ├─rewrite─────────▶ generator
                └────────────────┼────────────┴─need_evidence───▶ expert
"""

from langgraph.graph import END, START, StateGraph

from bluemed.agents.answer_generator import generator_node
from bluemed.agents.answer_reviewer import reviewer_node, route_after_review
from bluemed.agents.expert_agent import expert_node, route_after_expert, tools_node
from bluemed.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("expert", expert_node)
    graph.add_node("tools", tools_node)
    graph.add_node("generator", generator_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "expert")
    # Expert 的路由函数在"证据收集完"时返回 END；这里用映射表把它接到 generator，Expert 模块本身不用改
    graph.add_conditional_edges("expert", route_after_expert, {"tools": "tools", END: "generator"})
    graph.add_edge("tools", "expert")
    graph.add_edge("generator", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_review, ["generator", "expert", END])
    return graph.compile()
