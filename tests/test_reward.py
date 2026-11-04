"""奖励函数测试（内存图谱，不需要 Neo4j）。运行：python -m pytest tests/test_reward.py -q

除了边界情况，还把 RL_PLAN.md 第 4.3 节的每一种"钻空子"写成一个策略，检查它拿不到高分。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.env import ExpertEnv
from rl.make_questions import build
from rl.reward import FABRICATE, FORMAT_FAIL, evidence_entities, final_json, parse_final, score


@pytest.fixture(scope="module")
def dev():
    return build(seed=0)["dev"]


def _of(qs, t, **kw):
    return next(q for q in qs if q["type"] == t and all(q.get(k) == v for k, v in kw.items()))


def _answer(q):
    return None if "insufficient" in q["gold"] else q["gold"].get("answer", [])


def run(q, calls, answer):
    """执行一组调用，再用 answer 作为最后一行，返回 Reward。"""
    env = ExpertEnv()
    if calls:
        env.step(calls)
    return score(q, final_json(answer), env)


# ---------- 解析
@pytest.mark.parametrize("text", [None, "", "证据充分：找到了", '{"status": "ok"}', '{"status": "sufficient"}',
                                  '{"status": "sufficient", "answer": "鸡肝"}', "[1, 2]"])
def test_bad_format(text):
    assert parse_final(text) is None


def test_format_tolerates_prefix_text_and_code_fence():
    assert parse_final('查到了忌口食物。\n```json\n{"status": "sufficient", "answer": ["鸡肝", " 鸡心"]}\n```') \
        == {"status": "sufficient", "answer": ["鸡心", "鸡肝"]}


def test_format_fail_is_lowest(dev):
    q = _of(dev, "forward")
    r = score(q, "我觉得是鸡肝", ExpertEnv())
    assert r.total == FORMAT_FAIL and not r.format_ok


# ---------- 标准做法拿满分（校准：oracle + 标准答案 = 1）
def test_oracle_gets_full_marks(dev):
    for q in dev:
        r = run(q, q["oracle"], _answer(q))
        assert r.total == 1.0, (q["id"], r)


# ---------- 第 4.3 节的空子
def test_answer_without_querying_scores_zero(dev):
    """不查就答：答案和标准答案一字不差，但证据里没有 → 无据，算错。"""
    for t in ["forward", "backward", "symptoms", "two_hop"]:
        q = _of(dev, t)
        r = run(q, [], q["gold"]["answer"])
        assert r.content == 0 and r.n_ungrounded == len(q["gold"]["answer"]), t


def test_dump_everything_loses_precision(dev):
    """全列上去：把证据里所有实体都写进答案，分数低于标准答案。"""
    worse = 0
    qs = [q for q in dev if q["type"] in ("forward", "two_hop", "symptoms")]
    for q in qs:
        env = ExpertEnv()
        env.step(q["oracle"])
        r = score(q, final_json(sorted(evidence_entities(env))), env)
        assert r.total <= 1.0
        worse += r.total < 1.0
    assert worse == len(qs)                    # 证据里总会带上题目实体本身，所以一定不是满分


def test_always_insufficient(dev):
    """永远说查不到：能答的题 0 分；不查就说查不到也是 0 分。"""
    q = _of(dev, "forward")
    assert run(q, q["oracle"], None).total == 0
    u = _of(dev, "unanswerable")
    assert run(u, [], None).total == 0
    assert run(u, u["oracle"], None).total == 1


def test_food_pair_needs_checking(dev):
    """"能不能吃 Y"有能答的也有查不到的：只会说查不到、或只会把 Y 写上，都拿不到全部分数。"""
    yes = _of(dev, "forward", subtype="food_pair")
    no = _of(dev, "unanswerable", subtype="food_pair")
    assert run(yes, yes["oracle"], None).total == 0                       # 能答却说查不到
    assert run(yes, yes["oracle"], [yes["food"]]).total == 1
    assert run(no, no["oracle"], [no["food"]]).total == FABRICATE          # Y 不在证据里还写上
    assert run(no, no["oracle"], None).total == 1


def test_fabricate_on_unanswerable(dev):
    u = _of(dev, "unanswerable", subtype="food_pair")
    assert run(u, u["oracle"], ["可以吃"]).total == FABRICATE


def test_attribute_needs_the_right_field(dev):
    q = _of(dev, "attribute")
    d, f = q["gold"]["attribute"]
    other = "desc" if f != "desc" else "cause"
    assert run(q, [{"name": "get_disease_info", "args": {"disease": d, "fields": [other]}}], []).content == 0
    assert run(q, q["oracle"], None).total == 0              # 查到了却说查不到


def test_partial_answer_gets_partial_credit(dev):
    q = next(q for q in dev if q["type"] == "forward" and len(q["gold"]["answer"]) >= 2)
    r = run(q, q["oracle"], q["gold"]["answer"][:1])
    assert 0 < r.total < 1 and r.precision == 1 and r.recall < 1


# ---------- 扣分
def test_invalid_call_penalty(dev):
    q = _of(dev, "forward")
    bad = {"name": "kg_search", "args": {"entity": q["anchor"], "relation": "REQUIRES_DRUG"}}
    assert run(q, [bad] + q["oracle"], q["gold"]["answer"]).total == pytest.approx(0.9)


def test_extra_calls_penalty(dev):
    q = _of(dev, "forward", min_calls=1)
    rels = ["HAS_SYMPTOM", "REQUIRES_CHECK", "TREATED_BY", "ACCOMPANIES", "USES_TREATMENT"]
    extra = [{"name": "kg_search", "args": {"entity": q["anchor"], "relation": r}} for r in rels
             if r not in q["relations"]][:3]
    r = run(q, q["oracle"] + extra, q["gold"]["answer"])
    assert r.penalty == pytest.approx(0.05 * (1 + len(extra) - 3))      # 最少 1 次 + 免费 2 次


def test_penalty_is_capped(dev):
    q = _of(dev, "forward")
    bad = [{"name": "nope", "args": {}}] * 5
    r = run(q, bad, q["gold"]["answer"])
    assert r.penalty == 0.5 and r.total > FORMAT_FAIL


def test_reward_dict_is_json_serializable(dev):
    q = _of(dev, "two_hop")
    json.dumps(run(q, q["oracle"], q["gold"]["answer"]).to_dict())
