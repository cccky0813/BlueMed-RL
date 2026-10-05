"""Answer Reviewer 规则层的确定性测试（不调用大模型）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bluemed.agents.answer_reviewer import MAX_REVIEW_ROUNDS, route_after_review, rule_check
from bluemed.schemas import Answer, Evidence, Review

EV = {e.id: e for e in [Evidence.make("triple", "痛风", "SHOULD_NOT_EAT", "鸡肝")]}
EID = next(iter(EV))


def ans(text, cites, notice="", insufficient=False):
    return Answer(answer=text, citations=cites, safety_notice=notice, insufficient=insufficient)


def test_valid_answer_passes_rules():
    assert rule_check("痛风不能吃什么", ans(f"应避免鸡肝 [{EID}]", [EID]), EV) is None


def test_fabricated_citation_is_caught():
    r = rule_check("痛风不能吃什么", ans("应避免海鲜 [Edeadbe]", ["Edeadbe"]), EV)
    assert r.verdict == "rewrite" and "Edeadbe" in r.reason


def test_fabricated_inline_id_is_caught_even_if_citations_list_is_clean():
    r = rule_check("痛风不能吃什么", ans(f"应避免鸡肝 [{EID}]、啤酒 [Ebeef00]", [EID]), EV)
    assert r.verdict == "rewrite" and "Ebeef00" in r.reason


def test_answer_without_citation_is_rejected():
    assert rule_check("痛风不能吃什么", ans("应避免鸡肝", []), EV).verdict == "rewrite"


def test_honest_insufficient_answer_without_citation_is_allowed():
    assert rule_check("痛风能喝啤酒吗", ans("图谱中没有相关信息", [], insufficient=True), EV) is None


def test_drug_question_requires_safety_notice():
    assert rule_check("痛风吃什么药", ans(f"鸡肝 [{EID}]", [EID], notice=""), EV).verdict == "rewrite"
    assert rule_check("痛风吃什么药", ans(f"鸡肝 [{EID}]", [EID], notice="请遵医嘱"), EV) is None


def test_route_reads_next_step():
    assert route_after_review({"review": Review(verdict="rewrite", reason="x"), "next_step": "generator"}) == "generator"
    assert route_after_review({"review": Review(verdict="need_evidence", reason="x"), "next_step": "expert"}) == "expert"
    assert route_after_review({"review": Review(verdict="rewrite", reason="x"), "next_step": "end",
                               "review_rounds": MAX_REVIEW_ROUNDS}) == "__end__"
