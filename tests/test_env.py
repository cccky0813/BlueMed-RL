"""工具执行环境测试（内存图谱，不需要 Neo4j）。运行：python -m pytest tests/test_env.py -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bluemed.agents.expert_agent import DUPLICATE_CALL, LIMIT_REACHED, MAX_TOOL_CALLS
from rl.env import ExpertEnv

GOUT = {"name": "kg_search", "args": {"entity": "痛风", "relation": "SHOULD_NOT_EAT"}}


def test_normal_call_collects_evidence():
    env = ExpertEnv()
    [r] = env.step([GOUT])
    assert r.status == "ok" and "鸡肝" in r.content
    assert {e.object for e in env.evidence.values()} >= {"鸡肝"} and env.calls == 1


def test_duplicate_across_turns_is_blocked():
    env = ExpertEnv()
    env.step([GOUT])
    [r] = env.step([GOUT])
    assert r.status == "duplicate" and r.content == DUPLICATE_CALL and env.calls == 1


def test_duplicate_in_same_turn_is_executed_like_online():
    env = ExpertEnv()
    results = env.step([GOUT, GOUT])
    assert [r.status for r in results] == ["ok", "ok"] and env.calls == 2


def test_call_limit():
    env = ExpertEnv()
    calls = [{"name": "kg_search", "args": {"entity": d, "relation": "HAS_SYMPTOM"}}
             for d in ["痛风", "糖尿病", "高血压", "感冒", "肺炎", "哮喘"]]
    results = env.step(calls)
    assert [r.status for r in results] == ["ok"] * MAX_TOOL_CALLS + ["limit"]
    assert results[-1].content == LIMIT_REACHED and env.exhausted


def test_invalid_relation_is_reported_not_raised():
    env = ExpertEnv()
    [r] = env.step([{"name": "kg_search", "args": {"entity": "高血压", "relation": "REQUIRES_DRUG"}}])
    assert r.status == "invalid" and "参数不合法" in r.content and "relation" in r.content
    assert env.invalid == 1 and env.calls == 1 and not env.evidence


def test_unknown_tool_is_reported_not_raised():
    env = ExpertEnv()
    [r] = env.step([{"name": "search_web", "args": {"q": "痛风"}}])
    assert r.status == "invalid" and "kg_search" in r.content and env.invalid == 1


def test_all_kg_tools_run_on_memory_graph():
    env = ExpertEnv()
    results = env.step([
        {"name": "link_entity", "args": {"name": "糖尿病人"}},
        {"name": "get_disease_info", "args": {"disease": "糖尿病", "fields": ["cause"]}},
        {"name": "find_diseases_by_symptoms", "args": {"symptoms": ["头痛", "发烧"]}},
        {"name": "expand_multi_hop", "args": {"entity": "糖尿病", "question": "糖尿病的并发症要做什么检查"}},
    ])
    assert all(r.status == "ok" for r in results), [r.content for r in results]
    assert "「糖尿病」" in results[0].content
