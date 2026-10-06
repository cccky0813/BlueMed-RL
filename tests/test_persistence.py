"""记忆持久化测试：写入 → 关闭连接（模拟服务重启）→ 重新打开 → 读回。不调用大模型。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bluemed.graph import build_graph
from bluemed.memory.persistence import close_memory, open_memory
from bluemed.memory.profile import load_profile, save_profile
from bluemed.schemas import Answer, Evidence


def test_short_term_memory_survives_restart(tmp_path):
    db = tmp_path / "memory.db"
    config = {"configurable": {"thread_id": "t1", "user_id": "u1"}}
    e = Evidence.make("triple", "痛风", "SHOULD_NOT_EAT", "鸡肝")
    answer = Answer(answer=f"痛风忌吃鸡肝 [{e.id}]", citations=[e.id], safety_notice="", insufficient=False)

    saver, store = open_memory(db)
    graph = build_graph(saver, store)
    # 不跑大模型：以 remember 节点的身份直接写入一轮结果，走的是真实的 checkpoint 写入路径
    graph.update_state(config, {"conversation": [{"user": "痛风忌口", "assistant": answer.answer}],
                                "answer": answer, "evidence": {e.id: e}}, as_node="remember")
    close_memory(saver, store)                                    # 模拟服务重启

    saver, store = open_memory(db)
    state = build_graph(saver, store).get_state(config).values
    assert state["conversation"][0]["user"] == "痛风忌口"
    assert isinstance(state["answer"], Answer)                    # 自定义类型能按白名单还原
    assert state["evidence"][e.id].object == "鸡肝"
    other = build_graph(saver, store).get_state({"configurable": {"thread_id": "t2"}}).values
    assert other == {}                                            # 不同会话互相隔离
    close_memory(saver, store)


def test_long_term_profile_survives_restart(tmp_path):
    db = tmp_path / "memory.db"
    saver, store = open_memory(db)
    save_profile(store, "u1", {"subject": "父亲", "conditions": ["痛风"], "allergies": ["青霉素"]})
    close_memory(saver, store)

    saver, store = open_memory(db)
    assert load_profile(store, "u1") == {"subject": "父亲", "conditions": ["痛风"], "allergies": ["青霉素"]}
    assert load_profile(store, "u2")["conditions"] == []          # 不同用户互相隔离
    close_memory(saver, store)
