"""内存图谱和 Neo4j 的一致性测试。运行：python -m pytest tests/test_kg_memory.py -q（需要 Neo4j 在运行）

比较规则：Neo4j 的 LIMIT 没有 ORDER BY，超出上限时返回哪几条不确定。
所以没被截断时要求结果集合完全相同；被截断时只要求条数相同、且都属于完整结果。
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bluemed.kg import MAX_ROWS, Neo4jBackend, run_read
from bluemed.schemas import NODE_TYPES, RELATIONS
from bluemed.tools import kg_tools
from rl.kg_memory import CONTAINS_LIMIT, get_memory_kg

N = 200                                   # 每项随机抽多少个查询


@pytest.fixture(scope="module")
def kg():
    try:
        run_read("RETURN 1")
    except Exception as e:                # noqa: BLE001
        pytest.skip(f"Neo4j 不可用：{e}")
    return get_memory_kg()


@pytest.fixture(scope="module")
def neo():
    return Neo4jBackend()


def _same_or_truncated(neo_rows: list, full_rows: list, limit: int) -> bool:
    neo_set, full_set = set(neo_rows), set(full_rows)
    if len(full_set) < limit:
        return neo_set == full_set
    return len(neo_set) == limit and neo_set <= full_set


# ---------- 整张图
def test_graph_counts_match(kg):
    nodes, edges = kg.stats()
    assert nodes == {r["l"]: r["c"] for r in run_read("MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c")}
    assert edges == {r["t"]: r["c"] for r in run_read("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c")}


# ---------- 6 种查询
def test_relation_matches(kg, neo):
    rng = random.Random(0)
    for _ in range(N):
        rel = rng.choice(sorted(RELATIONS))
        if rng.random() < 0.5:
            name, direction, full = rng.choice(kg.diseases), "forward", None
            full = kg.targets(name, rel)
        else:
            name, direction = rng.choice(kg.names(kg_tools_target(rel))), "backward"
            full = kg.sources(name, rel)
        limit = rng.choice([5, 15, 30])
        neo_rows = [(r["s"], r["t"]) for r in neo.relation(name, rel, direction, limit)]
        full_rows = [(name, t) for t in full] if direction == "forward" else [(s, name) for s in full]
        assert _same_or_truncated(neo_rows, full_rows, limit), (name, rel, direction)
        assert [(r["s"], r["t"]) for r in kg.relation(name, rel, direction, limit)] == full_rows[:limit]


def kg_tools_target(rel: str) -> str:
    from rl.kg_memory import TARGET_LABEL
    return TARGET_LABEL[rel]


def test_properties_match(kg, neo):
    rng = random.Random(1)
    for d in rng.sample(kg.diseases, N):
        assert kg.properties(d) == neo.properties(d), d


def test_exact_and_contains_match(kg, neo):
    rng = random.Random(2)
    labels = sorted(NODE_TYPES)
    names = rng.sample(sorted(kg.labels), N)
    queries = names + [n[:-1] for n in names if len(n) > 2] + ["小孩" + n for n in names[:50]]
    for q in queries:
        e_neo, e_mem = neo.exact(q, labels), kg.exact(q, labels)
        assert (e_neo is None) == (e_mem is None), q
        if e_neo:
            assert e_neo["name"] == e_mem["name"] and e_neo["label"] in kg.labels[q]
        neo_rows = {(r["name"], r["label"]) for r in neo.contains(q, labels)}
        mem_rows = {(r["name"], r["label"]) for r in kg.contains(q, labels)}
        if len(neo_rows) < CONTAINS_LIMIT:
            assert neo_rows == mem_rows, q
        else:
            assert len(mem_rows) == CONTAINS_LIMIT


def test_neighbors_match(kg, neo):
    rng = random.Random(3)
    per_rel = 5
    for name in rng.sample(kg.diseases, N // 2) + rng.sample(kg.names("Symptom"), N // 2):
        neo_rows = neo.neighbors(name, per_rel)
        if len(neo_rows) == MAX_ROWS:                              # 被总行数上限截断，两边取的不一定一样
            continue
        # 按关系类型比较（不按方向）：同名的疾病节点和症状节点之间的边，Neo4j 会从两头各算一次，按名字分不出方向
        groups, mem_groups = defaultdict(list), defaultdict(list)
        for r in neo_rows:
            groups[r["rel"]].append((r["s"], r["t"], r["m"], r["label"]))
        for r in kg.neighbors(name, per_rel):
            mem_groups[r["rel"]].append((r["s"], r["t"], r["m"], r["label"]))
        assert groups.keys() == mem_groups.keys(), name
        for rel, rows in groups.items():
            n_out = len(kg.targets(name, rel))
            n_in = len([s for s in kg.sources(name, rel) if not (s == name and rel == "ACCOMPANIES")])   # 自环只算一次
            if n_out <= per_rel and n_in <= per_rel:
                assert sorted(rows) == sorted(mem_groups[rel]), (name, rel)
            else:
                assert len(rows) == len(mem_groups[rel]) == min(n_out, per_rel) + min(n_in, per_rel), (name, rel)


def test_diseases_by_symptoms_match(kg, neo):
    rng = random.Random(4)
    with_symptoms = [d for d in kg.diseases if len(kg.targets(d, "HAS_SYMPTOM")) >= 3]
    for d in rng.sample(with_symptoms, N):
        symptoms = rng.sample(kg.targets(d, "HAS_SYMPTOM"), rng.choice([1, 2, 3]))
        neo_rows = [(r["d"], frozenset(r["hit"])) for r in neo.diseases_by_symptoms(symptoms, 10)]
        mem_rows = [(r["d"], frozenset(r["hit"])) for r in kg.diseases_by_symptoms(symptoms, 10)]
        assert neo_rows == mem_rows, symptoms


# ---------- 5 个工具端到端（同一份工具代码，只换后端）
@pytest.fixture
def run_both(kg):
    def _run(tool, *args, **kwargs):
        kg_tools.set_backend(Neo4jBackend())
        neo_out = tool.func(*args, **kwargs)
        kg_tools.set_backend(kg)
        mem_out = tool.func(*args, **kwargs)
        return neo_out, mem_out
    yield _run
    kg_tools.set_backend(Neo4jBackend())


def _evidence_ids(out) -> set[str]:
    return {e.id for e in out[1]}


def test_tools_match(kg, run_both):
    rng = random.Random(5)
    diseases = rng.sample(kg.diseases, 60)
    for d in diseases:
        rel = rng.choice(sorted(RELATIONS))
        neo_out, mem_out = run_both(kg_tools.kg_search, d, rel)
        if len(kg.targets(d, rel)) < 15:                           # 默认 limit=15，没截断时文字也应一致（行序可能不同）
            assert sorted(neo_out[0].splitlines()) == sorted(mem_out[0].splitlines()), (d, rel)
        neo_out, mem_out = run_both(kg_tools.get_disease_info, d, ["cause", "prevent"])
        assert neo_out[0] == mem_out[0], d
        symptoms = kg.targets(d, "HAS_SYMPTOM")[:2]
        if symptoms:
            neo_out, mem_out = run_both(kg_tools.find_diseases_by_symptoms, symptoms)
            assert neo_out[0] == mem_out[0], symptoms
    for q in ["糖尿病人", "小孩中耳炎", "痛风发作", "头疼", "高血压病人"]:
        neo_out, mem_out = run_both(kg_tools.link_entity, q)
        assert neo_out[0] == mem_out[0], q


def test_expand_multi_hop_valid_and_deterministic(kg):
    """多跳先取邻居（每组最多 5 个）再按相似度挑选。邻居超过 5 个时 Neo4j 取哪几个不确定、没有规律
    （实测 Neo4j 的返回顺序既不是 json 顺序也不是倒序），所以这里不和 Neo4j 逐条比，
    只要求：内存版每次结果相同，且每条证据都是图里真实存在的边。"""
    kg_tools.set_backend(kg)
    try:
        rng = random.Random(6)
        for d in rng.sample(kg.diseases, 30):
            question = f"{d}的并发症要做什么检查"
            first = kg_tools.expand_multi_hop.func(d, question)
            assert first == kg_tools.expand_multi_hop.func(d, question)
            for e in first[1]:
                assert e.object in kg.targets(e.subject, e.predicate), e
    finally:
        kg_tools.set_backend(Neo4jBackend())
