"""Expert Agent 的确定性测试：不调用大模型，手动构造"模型请求调用工具"的消息，验证循环控制和防护机制。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from bluemed.agents.expert_agent import MAX_TOOL_CALLS, route_after_expert, tools_node
from bluemed.tools.kg_tools import find_diseases_by_symptoms


def ai_calls(*calls):
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(calls)])


def test_tools_node_executes_and_collects_evidence():
    state = {"messages": [HumanMessage("痛风不能吃什么"),
                          ai_calls(("kg_search", {"entity": "痛风", "relation": "SHOULD_NOT_EAT"}))]}
    out = tools_node(state)
    assert out["tool_calls"] == 1 and len(out["evidence"]) >= 4
    assert "鸡肝" in out["messages"][0].content


def test_repeated_call_is_not_executed_again():
    call = ("kg_search", {"entity": "痛风", "relation": "SHOULD_NOT_EAT"})
    first = tools_node({"messages": [HumanMessage("q"), ai_calls(call)]})
    state = {"messages": [HumanMessage("q"), ai_calls(call), *first["messages"], ai_calls(call)],
             "tool_calls": first["tool_calls"]}
    second = tools_node(state)
    assert second["tool_calls"] == 1                         # 计数没有增加
    assert "已经执行过" in second["messages"][0].content


def test_call_limit_blocks_extra_calls():
    state = {"messages": [HumanMessage("q"), ai_calls(("kg_search", {"entity": "痛风", "relation": "HAS_SYMPTOM"}))],
             "tool_calls": MAX_TOOL_CALLS}
    out = tools_node(state)
    assert out["tool_calls"] == MAX_TOOL_CALLS and "上限" in out["messages"][0].content


def test_route():
    assert route_after_expert({"messages": [ai_calls(("kg_search", {}))], "tool_calls": 0}) == "tools"
    assert route_after_expert({"messages": [AIMessage("证据充分：…")], "tool_calls": 1}) == END
    assert route_after_expert({"messages": [ai_calls(("kg_search", {}))], "tool_calls": MAX_TOOL_CALLS}) == END


def test_find_diseases_by_symptoms_ranks_full_match_first():
    content, evidence = find_diseases_by_symptoms.func(["头痛", "发烧"])
    assert content.splitlines()[0].startswith("命中 2/2") and "感冒" in content.splitlines()[0]
