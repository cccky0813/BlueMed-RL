"""② Expert Agent：医学知识专家，ReAct 循环（思考 → 调用工具 → 观察 → 再思考）。

图结构：
    expert ──有工具调用──▶ tools ──▶ expert（循环）
       └──不再调用工具 / 达到上限──▶ 结束
"""

import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from bluemed.llm import get_llm
from bluemed.state import AgentState
from bluemed.tools.kg_tools import KG_TOOLS

MAX_TOOL_CALLS = 5
TOOLS_BY_NAME = {t.name: t for t in KG_TOOLS}

EXPERT_PROMPT = f"""你是医学知识专家（Expert Agent），负责为用户问题从医疗知识图谱中收集证据，不负责写最终回答。

查询策略：
1. 先想清楚问题需要什么信息，选最直接的工具：
   - 疾病的症状、检查、用药、饮食、科室、治疗、并发症 → kg_search（forward）
   - 用户描述了多个症状，问可能是什么病 → find_diseases_by_symptoms（一次求交集，不要逐个疾病核对）
   - 已知单个症状 / 药物 / 检查，反查疾病 → kg_search（backward）
   - 病因、预防、治愈率、易感人群等 → get_disease_info（这些是属性，不是关系）
2. 先用一跳查询，只查问题直接问到的关系（问"忌口"就只查 SHOULD_NOT_EAT，不要顺带查宜吃、食谱）。
   一跳结果已经能回答问题时，立即停止，不要再扩展；只有问题本身需要间接知识（例如"并发症的检查"），
   或一跳确实查不到时，才用 expand_multi_hop。
3. 实体名不确定时可以先用 link_entity；kg_search 也会自动链接。
4. 不要用相同参数重复调用同一个工具；多个互不依赖的查询可以在同一轮里一起调用。
5. 工具调用总数不超过 {MAX_TOOL_CALLS} 次。
6. 证据足够时停止调用工具。最后只输出一行，不要写回答（回答由 Answer Agent 负责）：
   "证据充分：<一句话说明找到了什么>" 或 "证据不足：<缺什么>"。
只使用工具返回的信息，不要凭自己的知识补充。"""


def expert_node(state: AgentState) -> dict:
    """思考：模型看到问题和之前所有的工具结果，决定继续调用工具还是结束。"""
    # 次数用完时不再提供工具，模型只能根据已有证据写总结（否则可能停在"想调工具却被拒绝"的状态）
    llm = get_llm() if state.get("tool_calls", 0) >= MAX_TOOL_CALLS else get_llm().bind_tools(KG_TOOLS)
    first = [] if state.get("messages") else [("user", _first_message(state))]
    reply = llm.invoke([SystemMessage(EXPERT_PROMPT)] + (state.get("messages") or []) + first)
    return {"messages": first + [reply]}


def _first_message(state: AgentState) -> str:
    """本轮第一条消息：问题 + Parser（LoRA 模型）给的解析线索。线索只供参考，Expert 自己决定怎么查。"""
    parsed = state.get("parsed") or {}
    hint = ""
    if parsed.get("entities") or parsed.get("relations"):
        ents = "、".join(f"{e['name']}({e.get('type', '')})" for e in parsed.get("entities", []))
        rels = "、".join(r["type"] for r in parsed.get("relations", []))
        hint = f"\n（解析线索，仅供参考：实体 {ents or '无'}；关系 {rels or '未识别'}）"
    return state["question"] + hint


def tools_node(state: AgentState) -> dict:
    """行动：执行模型请求的所有工具调用，把结果（ToolMessage）和证据写回 State。"""
    last: AIMessage = state["messages"][-1]
    done = {(m.name, m.additional_kwargs.get("args_key")) for m in state["messages"] if isinstance(m, ToolMessage)}
    count = state.get("tool_calls", 0)
    results, evidence = [], {}

    for call in last.tool_calls:
        args_key = json.dumps(call["args"], ensure_ascii=False, sort_keys=True)
        if count >= MAX_TOOL_CALLS:                                   # 防护 1：调用次数上限
            content, artifact = f"已达到工具调用上限（{MAX_TOOL_CALLS} 次），请根据已有证据结束。", []
        elif (call["name"], args_key) in done:                        # 防护 2：重复调用检测
            content, artifact = "这个查询已经执行过了，结果见前文，请不要重复调用。", []
        else:
            msg = TOOLS_BY_NAME[call["name"]].invoke(call)            # 真正执行工具，返回带 artifact 的 ToolMessage
            content, artifact = msg.content, msg.artifact or []
            count += 1
        for e in (artifact if isinstance(artifact, list) else []):
            evidence[e.id] = e
        results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"],
                                   additional_kwargs={"args_key": args_key}))

    return {"messages": results, "evidence": evidence, "tool_calls": count}


def route_after_expert(state: AgentState) -> str:
    """条件边：模型请求了工具且未超过上限 → 去执行；否则结束。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls and state.get("tool_calls", 0) < MAX_TOOL_CALLS:
        return "tools"
    return END


def build_expert_graph():
    graph = StateGraph(AgentState)
    graph.add_node("expert", expert_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "expert")
    graph.add_conditional_edges("expert", route_after_expert, ["tools", END])
    graph.add_edge("tools", "expert")                                 # 执行完工具，回到 expert 继续思考（循环）
    return graph.compile()
