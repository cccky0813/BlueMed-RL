"""做题循环测试：用按剧本出牌的假模型，不需要真模型和 Neo4j。运行：python -m pytest tests/test_rollout.py -q"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bluemed.agents.expert_agent import MAX_TOOL_CALLS
from rl.make_questions import build
from rl.reward import FORMAT_FAIL, final_json
from rl.rollout import parse_qwen, rollout


@pytest.fixture(scope="module")
def dev():
    return build(seed=0)["dev"]


class Script:
    """按顺序返回预先写好的回复；记录每次是否提供了工具。"""
    def __init__(self, replies):
        self.replies, self.saw_tools = list(replies), []

    def __call__(self, messages, tools):
        self.saw_tools.append(tools is not None)
        return self.replies.pop(0) if self.replies else {"content": "", "tool_calls": []}


def test_oracle_script_scores_one(dev):
    q = next(q for q in dev if q["type"] == "two_hop")
    first, rest = q["oracle"][:1], q["oracle"][1:]           # 先查并发症，下一轮再查它们的检查
    p = Script([{"content": "", "tool_calls": [dict(c) for c in first]},
                {"content": "", "tool_calls": [dict(c) for c in rest]},
                {"content": final_json(q["gold"]["answer"]), "tool_calls": []}])
    t = rollout(q, p)
    assert t.reward.total == 1.0 and t.turns == 3
    roles = [m["role"] for m in t.messages]
    assert roles[:3] == ["system", "user", "assistant"] and roles.count("tool") == len(q["oracle"])
    call = t.messages[2]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == first[0]["args"]
    assert t.messages[3]["tool_call_id"] == call["id"]


def test_tools_removed_after_limit(dev):
    q = next(q for q in dev if q["type"] == "forward")
    many = [{"name": "kg_search", "args": {"entity": q["anchor"], "relation": r}}
            for r in ["HAS_SYMPTOM", "REQUIRES_CHECK", "TREATED_BY", "ACCOMPANIES", "USES_TREATMENT"]]
    p = Script([{"content": "", "tool_calls": many},
                {"content": final_json([]), "tool_calls": [{"name": "kg_search", "args": {}}]}])
    t = rollout(q, p)
    assert p.saw_tools == [True, False]                  # 用完 5 次后不再提供工具
    assert t.final is not None and t.reward.calls == MAX_TOOL_CALLS


def test_no_final_answer_is_format_fail(dev):
    q = dev[0]
    loop = {"content": "", "tool_calls": [{"name": "kg_search", "args": {"entity": q["anchor"], "relation": "HAS_SYMPTOM"}}]}
    t = rollout(q, Script([dict(loop, tool_calls=[dict(c) for c in loop["tool_calls"]]) for _ in range(20)]), max_turns=4)
    assert t.final is None and t.reward.total == FORMAT_FAIL
    assert t.tool_status == ["ok", "duplicate", "duplicate", "duplicate"]


def test_parse_qwen():
    text = ('先查一下。\n<tool_call>\n{"name": "kg_search", "arguments": {"entity": "痛风", "relation": "SHOULD_NOT_EAT"}}\n'
            '</tool_call>\n<tool_call>\n{{"name": "bad"}}\n</tool_call>')
    r = parse_qwen(text)
    assert r["content"] == "先查一下。"
    assert r["tool_calls"] == [{"name": "kg_search", "args": {"entity": "痛风", "relation": "SHOULD_NOT_EAT"}}]
