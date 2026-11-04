"""出题程序测试（内存图谱，不需要 Neo4j）。运行：python -m pytest tests/test_make_questions.py -q"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.make_questions import SIZES, TOOL_LIMIT, TYPES, build, is_dirty, verify


@pytest.fixture(scope="module")
def data():
    return build(seed=0)


def test_sizes_and_types(data):
    assert {s: len(qs) for s, qs in data.items()} == SIZES
    assert {q["type"] for q in data["test"]} == set(TYPES)
    assert all(sum(q["type"] == t for q in data["test"]) == SIZES["test"] // len(TYPES) for t in TYPES)


def test_reproducible(data):
    again = build(seed=0)
    assert [q["question"] for q in again["train"]] == [q["question"] for q in data["train"]]


def test_no_anchor_leak(data):
    anchors = {s: {q["anchor"] for q in qs} for s, qs in data.items()}
    assert not anchors["train"] & anchors["dev"] and not anchors["train"] & anchors["test"]
    assert not anchors["dev"] & anchors["test"]


def test_every_question_solvable_by_oracle(data):
    for q in data["dev"] + data["test"]:
        assert verify(q), q


def test_answers_fit_default_tool_limit(data):
    for q in data["train"]:
        if q["type"] in ("forward", "backward"):
            assert len(q["gold"]["answer"]) <= TOOL_LIMIT * len(q["relations"])
        assert q["min_calls"] == len(q["oracle"]) >= 1


def test_no_dirty_names(data):
    for q in data["train"]:
        assert not any(is_dirty(x) for x in [q["anchor"]] + q["gold"].get("answer", [])), q


def test_unanswerable_has_no_answer_in_oracle_evidence(data):
    for q in data["test"]:
        if q["type"] == "unanswerable":
            assert q["gold"] == {"insufficient": True}
