"""Phase 0 玩具图：分诊。急症问题走紧急提示，普通问题走正常回答。用来理解 LangGraph 的 4 个核心概念。"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 概念 1：State —— 所有节点共享的"黑板"，每个节点读它、往里写东西
class TriageState(TypedDict):
    question: str
    is_emergency: bool
    answer: str


# 概念 2：Node —— 一个普通函数：接收当前 State，返回要更新的字段
def check_emergency(state: TriageState) -> dict:
    keywords = ["胸痛", "喘不上气", "昏迷", "大出血", "抽搐","呼吸困难"]
    return {"is_emergency": any(k in state["question"] for k in keywords)}


def emergency_reply(state: TriageState) -> dict:
    return {"answer": "⚠️ 这可能是急症，请立即拨打 120 或前往最近的急诊。"}


def normal_reply(state: TriageState) -> dict:
    return {"answer": f"（正常流程）收到问题：{state['question']}，接下来交给 Agent 检索回答。"}

def out_of_scope_reply(state: TriageState) -> dict:
    return {"answer": "我只能回答医疗相关问题."}


# 概念 4：条件路由函数 —— 看 State，返回下一个节点的名字
def route(state: TriageState) -> str:
    if state["is_emergency"]:
        return "emergency_reply"
    if any(k in state["question"] for k in ["股票","天气"]):
        return  "out_of_scope_reply"
    return "normal_reply"


# 概念 3：Edge —— 把节点连起来，组成图
graph = StateGraph(TriageState)
graph.add_node("check_emergency", check_emergency)
graph.add_node("emergency_reply", emergency_reply)
graph.add_node("normal_reply", normal_reply)
graph.add_node("out_of_scope_reply", out_of_scope_reply)

graph.add_edge(START, "check_emergency")                      # 普通边：固定走向
graph.add_conditional_edges("check_emergency", route)          # 条件边：由 route() 决定走向
graph.add_edge("emergency_reply", END)
graph.add_edge("normal_reply", END)
graph.add_edge("out_of_scope_reply", END)

app = graph.compile()                                          # 编译成可以运行的应用


if __name__ == "__main__":
    for q in ["痛风不能吃什么？", "我爸突然胸痛，喘不上气", "我妈呼吸困难", "明天天气怎么样", "天气太热我爸胸痛"]:
        print(f"\n问题：{q}")
        # stream 会在每个节点执行完后吐出一次结果，可以看到图是怎么一步步走的
        for step in app.stream({"question": q}):
            print("  执行节点 →", step)
        print("  最终结果：", app.invoke({"question": q})["answer"])
