"""Neo4j 访问层。所有工具只能通过 run_read() 查询图谱，安全限制集中在这里：
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
