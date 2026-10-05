"""KG 工具单元测试。运行：python -m pytest tests -q（需要 Neo4j 在运行）"""

import sys
from pathlib import Path

import pytest
from neo4j.exceptions import ClientError
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bluemed.kg import run_read
from bluemed.schemas import Evidence
from bluemed.tools.kg_tools import _link, expand_multi_hop, get_disease_info, kg_search


# ---------- 实体链接
def test_link_exact():
    assert _link("痛风").method == "exact"


def test_link_fuzzy():
    r = _link("糖尿病人", "Disease")
    assert r.matched == "糖尿病" and r.method == "fuzzy"


def test_link_none_for_unrelated_wording():
    assert _link("肚子疼", "Symptom").matched is None       # 已知局限：同义词无字面重叠


# ---------- kg_search
def test_search_forward():
    content, evidence = kg_search.func("痛风", "SHOULD_NOT_EAT")
    assert "鸡肝" in {e.object for e in evidence}
    assert all(e.predicate == "SHOULD_NOT_EAT" for e in evidence)


def test_search_backward():
    _, evidence = kg_search.func("头痛", "HAS_SYMPTOM", direction="backward")
    assert evidence and all(e.object == "头痛" for e in evidence)


def test_search_auto_link():
    content, evidence = kg_search.func("糖尿病人", "REQUIRES_CHECK")
    assert "链接为「糖尿病」" in content and evidence


def test_search_rejects_invalid_relation():
    with pytest.raises(ValidationError):
        kg_search.invoke({"entity": "痛风", "relation": "DROP_DATABASE"})


def test_search_rejects_too_large_limit():
    with pytest.raises(ValidationError):
        kg_search.invoke({"entity": "痛风", "relation": "HAS_SYMPTOM", "limit": 1000})


# ---------- get_disease_info
def test_disease_info():
    _, evidence = get_disease_info.func("糖尿病", ["cause"])
    assert evidence and evidence[0].kind == "attribute" and evidence[0].predicate == "cause"


# ---------- expand_multi_hop
def test_multi_hop_has_second_hop_and_no_department_expansion():
    _, evidence = expand_multi_hop.func("糖尿病", "糖尿病的并发症需要做什么检查")
    assert any(e.hop == 2 for e in evidence)
    departments = {r["n"] for r in run_read("MATCH (n:Department) RETURN n.name AS n")}
    hop1_nodes = {e.object if e.subject == "糖尿病" else e.subject for e in evidence if e.hop == 1}
    assert not (hop1_nodes & departments)                   # 【修复 v1 问题 3】不从科室节点扩展


# ---------- 工具层安全
def test_read_only_transaction_blocks_writes():
    # 只接受"数据库拒绝写操作"这一种错误；写成 pytest.raises(Exception) 的话，任何 bug 都会让测试误通过
    with pytest.raises(ClientError, match="(?i)write"):
        run_read("CREATE (n:ShouldNotExist {name: 'x'}) RETURN n")
    assert not run_read("MATCH (n:ShouldNotExist) RETURN n")


def test_evidence_id_is_deterministic():
    a = Evidence.make("triple", "痛风", "SHOULD_NOT_EAT", "鸡肝")
    b = Evidence.make("triple", "痛风", "SHOULD_NOT_EAT", "鸡肝")
    assert a.id == b.id


def test_multi_hop_follows_relation_in_question():
    # 回归测试：问"并发症的检查"时，第一跳应选并发症（ACCOMPANIES），第二跳应出现检查（REQUIRES_CHECK）
    _, evidence = expand_multi_hop.func("糖尿病", "糖尿病的并发症需要做什么检查？")
    hop1 = [e for e in evidence if e.hop == 1]
    hop2 = [e for e in evidence if e.hop == 2]
    assert sum(e.predicate == "ACCOMPANIES" for e in hop1) >= len(hop1) - 1
    assert sum(e.predicate == "REQUIRES_CHECK" for e in hop2) >= 5
