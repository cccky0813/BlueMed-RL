"""难测评集的校验逻辑：假 DeepSeek + 假实体链接，不需要网络。运行：python -m pytest tests/test_hard_set.py -q"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bluemed.schemas import LinkResult
from rl.hard_set import Guess, Lay, make_distractor, make_free, make_lay
from rl.make_questions import build
from rl.rewrite import Mention, Rewrite

ALIASES = {"老胃病": "慢性胃炎"}


NODES = {"慢性胃炎", "肠胃炎"}                  # 假图谱里真有的实体（精确匹配）


def link(name, node_type=None):
    return LinkResult(query=name, matched=ALIASES.get(name, name), method="exact" if name in NODES else "fuzzy")


@pytest.fixture(scope="module")
def q():
    return dict(next(x for x in build(seed=0)["test"] if x["type"] == "attribute"),
                anchor="慢性胃炎", question="慢性胃炎是什么原因引起的？")


def script(**by_schema):
    """按 schema 依次返回预设回复。"""
    queues = {k: list(v) for k, v in by_schema.items()}
    return lambda schema, prompt: queues[schema.__name__].pop(0)


def R(text, *pairs):
    return Rewrite(question=text, mentions=[Mention(original=o, said=s) for o, s in pairs])


def test_free_keeps_entity_verbatim(q):
    row, _ = make_free(q, script(Rewrite=[R("慢性胃炎到底咋得的？", ("慢性胃炎", "慢性胃炎"))]), link, random.Random(0))
    assert row["question"] == "慢性胃炎到底咋得的？"
    row, reasons = make_free(q, script(Rewrite=[R("胃炎咋得的？", ("慢性胃炎", "胃炎"))] * 2), link, random.Random(0))
    assert row is None and "对不上" in reasons[0]


def test_lay_needs_unlinkable_but_guessable(q):
    rng = random.Random(0)
    # 俗称直接能链接回去：不算难
    row, why = make_lay(q, script(Lay=[Lay(has_lay=True, lay="老胃病")]), link, rng)
    assert row is None and "直接就能链接" in why[0]
    # 猜回来不是原实体：不可解
    row, why = make_lay(q, script(Lay=[Lay(has_lay=True, lay="胃不舒服")], Guess=[Guess(name="消化不良")]), link, rng)
    assert row is None and "不可解" in why[0]
    # 合格：俗称链接不回去、能猜回来、问题里只有俗称
    row, _ = make_lay(q, script(Lay=[Lay(has_lay=True, lay="胃一直发炎")], Guess=[Guess(name="慢性胃炎")],
                                Rewrite=[R("胃一直发炎是啥原因？")]), link, rng)
    assert row and row["lay"] == {"慢性胃炎": "胃一直发炎"}
    row, why = make_lay(q, script(Lay=[Lay(has_lay=True, lay="胃一直发炎")], Guess=[Guess(name="慢性胃炎")],
                                  Rewrite=[R("胃一直发炎，也就是慢性胃炎，是啥原因？")] * 2), link, rng)
    assert row is None and "不能写" in why[-1]


def test_lay_rejects_term_that_is_another_entity(q):
    row, why = make_lay(q, script(Lay=[Lay(has_lay=True, lay="肠胃炎")]), link, random.Random(0))
    assert row is None and "另一个实体" in why[0]


def test_lay_skips_terms_without_common_name(q):
    row, why = make_lay(q, script(Lay=[Lay(has_lay=False)]), link, random.Random(0))
    assert row is None and why == ["没有常见俗称"]


def test_distractor_must_mention_both(q):
    rng = random.Random(0)
    ok = R("我爸有高血压，我自己慢性胃炎是啥原因引起的？", ("慢性胃炎", "慢性胃炎"), ("高血压", "高血压"))
    row, _ = make_distractor(q, script(Rewrite=[ok]), link, rng, ["高血压"])
    assert row["distractor"] == "高血压"
    missing = R("慢性胃炎是啥原因引起的？", ("慢性胃炎", "慢性胃炎"))
    row, why = make_distractor(q, script(Rewrite=[missing, missing]), link, rng, ["高血压"])
    assert row is None and "高血压" in why[0]
