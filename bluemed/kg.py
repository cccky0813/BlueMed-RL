"""Neo4j 访问层。所有工具只能通过 run_read()（经由 Neo4jBackend）查询图谱，安全限制集中在这里：
1. 只读事务：即使查询语句里有写操作，数据库也会拒绝执行
2. 超时：单次查询超过 QUERY_TIMEOUT_S 秒自动中止
3. 条数上限：最多返回 MAX_ROWS 行，防止结果撑爆模型上下文
"""

from functools import lru_cache

from neo4j import GraphDatabase, unit_of_work

from bluemed import config

QUERY_TIMEOUT_S = 5
MAX_ROWS = 100


@lru_cache(maxsize=1)
def get_driver():
    """整个进程共用一个连接池（单例）。"""
    return GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))


def run_read(cypher: str, **params) -> list[dict]:
    """在只读事务中执行查询，返回字典列表。"""
    @unit_of_work(timeout=QUERY_TIMEOUT_S)          # 事务超时：超过 5 秒由数据库中止
    def _work(tx):
        result = tx.run(cypher, **params)
        return [record.data() for record in result][:MAX_ROWS]

    with get_driver().session() as session:
        return session.execute_read(_work)


class Neo4jBackend:
    """图谱工具用到的 6 种查询（Neo4j 实现）。

    工具的逻辑（实体链接、格式化、证据）只写一份，在 tools/kg_tools.py；
    查询换成内存实现（rl/kg_memory.py）就能在 RL 训练时脱离数据库，模型看到的工具返回逐字相同。
    注意：这里的 LIMIT 没有 ORDER BY，超出上限时返回哪几条由数据库决定（不保证顺序）。
    """

    def exact(self, name: str, labels: list[str]) -> dict | None:
        """名字完全相同的节点：{name, label}。"""
        rows = run_read("MATCH (n {name: $name}) WHERE any(l IN labels(n) WHERE l IN $labels) "
                        "RETURN n.name AS name, labels(n)[0] AS label LIMIT 1", name=name, labels=labels)
        return rows[0] if rows else None

    def contains(self, name: str, labels: list[str]) -> list[dict]:
        """名字和 name 互相包含的节点（名字至少 2 个字），最多 50 个：[{name, label}]。"""
        return run_read("MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels) AND size(n.name) >= 2 "
                        "AND (n.name CONTAINS $name OR $name CONTAINS n.name) "
                        "RETURN n.name AS name, labels(n)[0] AS label LIMIT 50", name=name, labels=labels)

    def relation(self, name: str, relation: str, direction: str, limit: int) -> list[dict]:
        """一跳关系：[{s, t}]。relation 由调用方按白名单校验过（要拼进 Cypher）。"""
        if direction == "forward":
            cypher = f"MATCH (s {{name: $name}})-[:{relation}]->(t) RETURN s.name AS s, t.name AS t LIMIT $limit"
        else:
            cypher = f"MATCH (s)-[:{relation}]->(t {{name: $name}}) RETURN s.name AS s, t.name AS t LIMIT $limit"
        return run_read(cypher, name=name, limit=limit)

    def properties(self, disease: str) -> dict:
        rows = run_read("MATCH (d:Disease {name: $name}) RETURN properties(d) AS p", name=disease)
        return rows[0]["p"] if rows else {}

    def neighbors(self, name: str, per_rel: int) -> list[dict]:
        """所有邻居，按（关系类型, 方向）分组、每组最多 per_rel 条：[{rel, s, t, m, label}]，m 是邻居。"""
        # 【修复 v1 问题 1】v1 先 LIMIT 10 再在 Python 里分组，导致只拿到存储顺序靠前的类型；
        # 这里在 Cypher 里先按关系类型分组，每种类型各取 per_rel 条
        # 同时按方向分组：同一种关系的出边和入边含义不同（糖尿病→尿毒症 是"糖尿病的并发症"，代谢综合征→糖尿病 则相反）
        return run_read(
            "MATCH (a {name: $name})-[r]-(m) "
            "WITH type(r) AS rel, startNode(r) = a AS outgoing, "
            "     collect({s: startNode(r).name, t: endNode(r).name, m: m.name, label: labels(m)[0]})[..$k] AS items "
            "UNWIND items AS it RETURN rel, it.s AS s, it.t AS t, it.m AS m, it.label AS label",
            name=name, k=per_rel)

    def diseases_by_symptoms(self, symptoms: list[str], limit: int) -> list[dict]:
        """有这些症状的疾病，按命中个数从多到少、再按名字排序：[{d, hit}]。"""
        return run_read(
            "UNWIND $names AS s MATCH (d:Disease)-[:HAS_SYMPTOM]->(x {name: s}) "
            "WITH d, collect(DISTINCT x.name) AS hit RETURN d.name AS d, hit ORDER BY size(hit) DESC, d.name LIMIT $limit",
            names=symptoms, limit=limit)
