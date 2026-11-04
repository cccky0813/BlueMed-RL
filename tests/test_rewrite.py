"""口语改写的校验逻辑测试：用假的 DeepSeek 和假的实体链接，不需要网络和 m3e。运行：python -m pytest tests/test_rewrite.py -q"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bluemed.schemas import LinkResult
from rl.make_questions import build
from rl.rewrite import Mention, Rewrite, key_entities, rewrite_one

ALIASES = {"老胃病": "慢性胃炎"}          # 假链接：精确名 + 这一个俗称


def fake_link(name, node_type=None):
    return LinkResult(query=name, matched=ALIASES.get(name, name))


@pytest.fixture(scope="module")
def dev():
    return build(seed=0)["dev"]


def _of(qs, t, **kw):
    return next(q for q in qs if q["type"] == t and all(q.get(k) == v for k, v in kw.items()))


def scripted(*replies):
    replies = list(replies)
    return lambda prompt: replies.pop(0)


def test_key_entities(dev):
    s = _of(dev, "symptoms")
    assert [n for n, _ in key_entities(s)] == s["symptoms"]
    f = _of(dev, "forward", subtype="food_pair")
    assert key_entities(f) == [(f["anchor"], "Disease"), (f["food"], "Food")]
    b = _of(dev, "backward")
    assert key_entities(b)[0][0] == b["anchor"] and key_entities(b)[0][1] != "Disease"


def test_keeps_answer_and_oracle(dev):
    q = _of(dev, "two_hop")
    d = q["anchor"]
    row, reasons = rewrite_one(q, scripted(Rewrite(question=f"我爸{d}好几年了，并发症要查啥？",
                                                   mentions=[Mention(original=d, said=d)])), fake_link)
    assert reasons == [] and row["id"] == q["id"] + "-oral" and row["split"] == "dev_oral"
    assert row["template"] == q["question"] and row["gold"] == q["gold"] and row["oracle"] == q["oracle"]


def test_retry_then_drop(dev):
    q = _of(dev, "forward", subtype="food_pair")
    d, y = q["anchor"], q["food"]
    bad_link = Rewrite(question=f"那个病能吃{y}吗", mentions=[Mention(original=d, said="那个病"),
                                                          Mention(original=y, said=y)])
    not_in_text = Rewrite(question=f"能吃{y}吗", mentions=[Mention(original=d, said=d), Mention(original=y, said=y)])
    row, reasons = rewrite_one(q, scripted(bad_link, not_in_text), fake_link)
    assert row is None and len(reasons) == 2
    assert "找不到" in reasons[0] and "没有原样出现" in reasons[1]


def test_alias_that_links_back_is_ok(dev):
    q = dict(_of(dev, "attribute"), anchor="慢性胃炎", question="慢性胃炎是什么原因引起的？")
    row, _ = rewrite_one(q, scripted(Rewrite(question="老胃病到底是咋得的啊？",
                                             mentions=[Mention(original="慢性胃炎", said="老胃病")])), fake_link)
    assert row and row["mentions"] == {"慢性胃炎": "老胃病"}


def test_missing_mention_and_llm_error(dev):
    q = _of(dev, "symptoms")
    def boom(prompt):
        raise ValueError("bad json")
    row, reasons = rewrite_one(q, boom, fake_link)
    assert row is None and reasons[0].startswith("输出格式错误")
    only_one = Rewrite(question=q["symptoms"][0], mentions=[Mention(original=q["symptoms"][0], said=q["symptoms"][0])])
    row, reasons = rewrite_one(q, scripted(only_one, only_one), fake_link)
    assert row is None and "缺少" in reasons[0]


def test_stock_phrases_rejected(dev):
    q = _of(dev, "attribute")
    d = q["anchor"]
    row, reasons = rewrite_one(q, scripted(Rewrite(question=f"{d}咋回事啊？急死人了！",
                                                   mentions=[Mention(original=d, said=d)]),
                                           Rewrite(question=f"{d}是咋回事？", mentions=[Mention(original=d, said=d)])),
                               fake_link)
    assert row and row["question"] == f"{d}是咋回事？" and "不要用" in reasons[0]


def test_split_question_rejected(dev):
    q = _of(dev, "two_hop", subtype="symptom")
    d = q["anchor"]
    two = Rewrite(question=f"{d}会有哪些并发症？这些并发症有啥症状？", mentions=[Mention(original=d, said=d)])
    one = Rewrite(question=f"{d}的并发症都有啥症状？", mentions=[Mention(original=d, said=d)])
    row, reasons = rewrite_one(q, scripted(two, one), fake_link)
    assert row and "一个问题" in reasons[0]
