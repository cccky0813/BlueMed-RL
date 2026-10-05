"""Expert Agent 的知识图谱工具。

设计原则：
- 粒度适中：4 个工具各管一件事，由 Agent 按需组合（简单问题一次 kg_search，复杂问题再 expand_multi_hop）
- 参数约束：Pydantic schema，关系类型是枚举，条数有上限；非法参数在执行前就被拒绝
- 返回两份：content（简短文字，给模型看）+ artifact（结构化证据，给程序用，后续 Reviewer 校验引用）
- 错误对模型友好：查不到时不抛异常，而是告诉模型下一步可以怎么做
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from bluemed.embedding import similarity
from bluemed.kg import run_read
from bluemed.schemas import (FIELD_CN, NODE_TYPES, RELATION_CN, RELATIONS, DiseaseField, Evidence,
                             LinkResult, NodeType, Relation)

LINK_THRESHOLD = 0.8
EXPANDABLE = ["Disease", "Symptom"]       # 多跳只从疾病和症状扩展


# ---------------------------------------------------------------- link_entity
def _link(name: str, node_type: str | None = None) -> LinkResult:
    labels = [node_type] if node_type in NODE_TYPES else sorted(NODE_TYPES)
    exact = run_read("MATCH (n {name: $name}) WHERE any(l IN labels(n) WHERE l IN $labels) "
                     "RETURN n.name AS name, labels(n)[0] AS label LIMIT 1", name=name, labels=labels)
    if exact:
        return LinkResult(query=name, matched=exact[0]["name"], node_type=exact[0]["label"], score=1.0, method="exact")
    if len(name) < 2:
        return LinkResult(query=name, matched=None)
    rows = run_read("MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels) AND size(n.name) >= 2 "
                    "AND (n.name CONTAINS $name OR $name CONTAINS n.name) "
                    "RETURN n.name AS name, labels(n)[0] AS label LIMIT 50", name=name, labels=labels)
    if not rows:
        return LinkResult(query=name, matched=None)
    scores = similarity(name, [r["name"] for r in rows])
    best = max(range(len(rows)), key=lambda i: scores[i])
    if scores[best] < LINK_THRESHOLD:
        return LinkResult(query=name, matched=None, score=scores[best])
    return LinkResult(query=name, matched=rows[best]["name"], node_type=rows[best]["label"],
                      score=round(scores[best], 3), method="fuzzy")


class LinkArgs(BaseModel):
    name: str = Field(description="用户问题中提到的医学实体名称，如 糖尿病人、小孩中耳炎")
    node_type: NodeType | None = Field(None, description="实体类型，已知时填写可提高准确率")


@tool(args_schema=LinkArgs, response_format="content_and_artifact")
def link_entity(name: str, node_type: str | None = None):
    """把用户说的实体名映射到知识图谱中的规范名称。名称不确定（口语化、带修饰词）时先调用它。"""
    r = _link(name, node_type)
    if r.matched is None:
        return f"图谱中没有找到与「{name}」匹配的实体，可以换一个更常见的说法再试。", r
    return f"「{name}」→ 图谱实体「{r.matched}」（{r.node_type}，{r.method}，相似度 {r.score}）", r


# ---------------------------------------------------------------- kg_search
class SearchArgs(BaseModel):
    entity: str = Field(description="已知的实体名，如 痛风、头痛")
    relation: Relation = Field(description="要查询的关系类型")
    direction: str = Field("forward", pattern="^(forward|backward)$",
                           description="forward：已知疾病查目标（痛风不能吃什么）；backward：已知目标反查疾病（头痛是什么病）")
    limit: int = Field(15, ge=1, le=30, description="最多返回条数")


@tool(args_schema=SearchArgs, response_format="content_and_artifact")
def kg_search(entity: str, relation: str, direction: str = "forward", limit: int = 15):
    """按关系类型在医疗知识图谱中精确查询，是回答大多数问题的主力工具。实体名不在图谱中时会自动做实体链接。"""
    if relation not in RELATIONS:                    # 白名单：关系类型要拼进 Cypher，必须再校验一次（纵深防御）
        return f"不支持的关系类型 {relation}。", []
    link = _link(entity)
    if link.matched is None:
        return f"图谱中没有「{entity}」，请先用 link_entity 换个说法确认实体。", []
    if direction == "forward":
        cypher = f"MATCH (s {{name: $name}})-[:{relation}]->(t) RETURN s.name AS s, t.name AS t LIMIT $limit"
    else:
        cypher = f"MATCH (s)-[:{relation}]->(t {{name: $name}}) RETURN s.name AS s, t.name AS t LIMIT $limit"
    rows = run_read(cypher, name=link.matched, limit=limit)

    note = f"（已将「{entity}」链接为「{link.matched}」）" if link.matched != entity else ""
    if not rows:
        return f"{note}「{link.matched}」没有 {relation}（{RELATION_CN[relation]}）类型的关系，可以换一种关系或用 expand_multi_hop 扩展。", []
    evidence = [Evidence.make("triple", r["s"], relation, r["t"]) for r in rows]
    # 明确告诉模型结果是否完整：否则只返回 4 条时，模型会怀疑被截断而加大 limit 重查一次
    total = "（已返回全部结果）" if len(rows) < limit else f"（只显示前 {limit} 条，可能还有更多）"
    return note + "\n".join(e.to_text() for e in evidence) + f"\n共 {len(rows)} 条{total}", evidence


# ---------------------------------------------------------------- get_disease_info
class InfoArgs(BaseModel):
    disease: str = Field(description="疾病名称")
    fields: list[DiseaseField] = Field(description="要查询的属性：desc 简介、cause 病因、prevent 预防、easy_get 易感人群、"
                                                   "get_way 传播途径、cure_lasttime 治疗周期、cured_prob 治愈率、cost_money 费用",
                                       min_length=1, max_length=4)


@tool(args_schema=InfoArgs, response_format="content_and_artifact")
def get_disease_info(disease: str, fields: list[str]):
    """查询疾病的文字属性（病因、预防、治愈率等）。这类信息存在疾病节点上，不是关系，不要用 kg_search 查。"""
    link = _link(disease, "Disease")
    if link.matched is None:
        return f"图谱中没有疾病「{disease}」。", []
    rows = run_read("MATCH (d:Disease {name: $name}) RETURN properties(d) AS p", name=link.matched)
    props = rows[0]["p"] if rows else {}
    evidence = [Evidence.make("attribute", link.matched, f, str(props[f])[:200]) for f in fields if props.get(f)]
    if not evidence:
        return f"「{link.matched}」没有记录这些属性：{'、'.join(FIELD_CN[f] for f in fields)}。", []
    return "\n".join(e.to_text() for e in evidence), evidence


# ---------------------------------------------------------------- expand_multi_hop
class ExpandArgs(BaseModel):
    entity: str = Field(description="起点实体")
    question: str = Field(description="用户的原始问题，用来判断哪些邻居与问题相关")
    top_k: int = Field(5, ge=1, le=8, description="选几个最相关的邻居做第二跳")
    per_node: int = Field(3, ge=1, le=5, description="每个邻居最多扩展几条")


def _neighbors(name: str, per_rel: int) -> list[dict]:
    # 【修复 v1 问题 1】v1 先 LIMIT 10 再在 Python 里分组，导致只拿到存储顺序靠前的类型；
    # 这里在 Cypher 里先按关系类型分组，每种类型各取 per_rel 条
    # 同时按方向分组：同一种关系的出边和入边含义不同（糖尿病→尿毒症 是"糖尿病的并发症"，代谢综合征→糖尿病 则相反）
    return run_read(
        "MATCH (a {name: $name})-[r]-(m) "
        "WITH type(r) AS rel, startNode(r) = a AS outgoing, "
        "     collect({s: startNode(r).name, t: endNode(r).name, m: m.name, label: labels(m)[0]})[..$k] AS items "
        "UNWIND items AS it RETURN rel, it.s AS s, it.t AS t, it.m AS m, it.label AS label",
        name=name, k=per_rel)


def _describe(r: dict) -> str:
    """把一条边转成中文短句用于和问题算相似度，例如"糖尿病 并发症 尿毒症"。
    只用邻居名字打分时，问题里的"并发症""检查"等关系信息完全没被用上。"""
    return f"{r['s']} {RELATION_CN.get(r['rel'], r['rel'])} {r['t']}"


@tool(args_schema=ExpandArgs, response_format="content_and_artifact")
def expand_multi_hop(entity: str, question: str, top_k: int = 5, per_node: int = 3):
    """从实体出发做两跳扩展，用于一跳证据不足、需要间接知识的问题（如：糖尿病的并发症要做什么检查）。
    会按与问题的相关度选出最相关的邻居再扩展，避免邻居数量爆炸。"""
    link = _link(entity)
    if link.matched is None:
        return f"图谱中没有「{entity}」。", []
    seed = link.matched
    hop1 = _neighbors(seed, per_rel=5)

    # 【修复 v1 问题 3】只从疾病 / 症状节点扩展，不从科室、分类节点扩展（否则会引入大量同科室的无关疾病）
    candidates = list({r["m"]: r for r in hop1 if r["label"] in EXPANDABLE and r["m"] != seed}.values())
    # 【修复 v1 问题 2】v1 用"实体名 + 英文关系名"拼接的字符串算向量；这里直接用用户的原始问题
    scores = similarity(question, [_describe(c) for c in candidates])
    chosen = sorted(zip(scores, candidates), key=lambda x: -x[0])[:top_k]

    evidence: dict[str, Evidence] = {}
    for score, c in chosen:
        e1 = Evidence.make("triple", c["s"], c["rel"], c["t"], hop=1, score=round(score, 3))
        evidence[e1.id] = e1
        hop2 = [r for r in _neighbors(c["m"], per_rel=per_node) if seed not in (r["s"], r["t"])]
        hop2_scores = similarity(question, [_describe(r) for r in hop2])
        for s2, r in sorted(zip(hop2_scores, hop2), key=lambda x: -x[0])[:per_node]:
            e2 = Evidence.make("triple", r["s"], r["rel"], r["t"], hop=2, score=round(s2, 3))
            evidence.setdefault(e2.id, e2)

    items = sorted(evidence.values(), key=lambda e: (e.hop, -e.score))[:20]
    if not items:
        return f"「{seed}」周围没有可扩展的疾病或症状节点。", []
    return "\n".join(f"{e.to_text()}（第{e.hop}跳）" for e in items), items


# ---------------------------------------------------------------- find_diseases_by_symptoms
class SymptomsArgs(BaseModel):
    symptoms: list[str] = Field(description="用户描述的症状，如 [\"头痛\", \"发烧\"]", min_length=1, max_length=6)
    limit: int = Field(10, ge=1, le=20, description="最多返回几个疾病")


@tool(args_schema=SymptomsArgs, response_format="content_and_artifact")
def find_diseases_by_symptoms(symptoms: list[str], limit: int = 10):
    """根据多个症状反查疾病，按命中的症状数量从多到少排序（同时具备的症状越多越靠前）。
    用户描述了两个及以上症状、问"可能是什么病"时，优先用它，不要逐个疾病去核对。"""
    # 设计动机：Phase 2 中 Agent 用"分别反查每个症状 + 逐个疾病核对"来求交集，5 次调用都不够、结果靠运气；
    # 求交集是确定性操作，一条 Cypher 就能完成，于是封装成工具
    linked = {s: _link(s, "Symptom").matched or _link(s).matched for s in symptoms}
    names = [n for n in linked.values() if n]
    if not names:
        return f"图谱中没有找到这些症状：{'、'.join(symptoms)}。", []
    rows = run_read(
        "UNWIND $names AS s MATCH (d:Disease)-[:HAS_SYMPTOM]->(x {name: s}) "
        "WITH d, collect(DISTINCT x.name) AS hit RETURN d.name AS d, hit ORDER BY size(hit) DESC, d.name LIMIT $limit",
        names=names, limit=limit)
    evidence = [Evidence.make("triple", r["d"], "HAS_SYMPTOM", s, score=len(r["hit"]) / len(names))
                for r in rows for s in r["hit"]]
    mapping = "、".join(f"{k}→{v}" for k, v in linked.items() if v and v != k)
    lines = [f"命中 {len(r['hit'])}/{len(names)} 个症状：{r['d']}（{'、'.join(r['hit'])}）" for r in rows]
    head = f"（症状链接：{mapping}）\n" if mapping else ""
    return head + "\n".join(lines) + "\n" + "\n".join(e.to_text() for e in evidence), evidence


KG_TOOLS = [link_entity, kg_search, get_disease_info, expand_multi_hop, find_diseases_by_symptoms]
