"""记忆持久化：短期记忆（checkpointer）和长期记忆（store）都落到同一个本地 SQLite 文件。

  短期记忆 SqliteSaver：每个 super-step 结束把整个 State 按 thread_id 存一个 checkpoint → 服务重启后接着聊
  长期记忆 SqliteStore：健康档案按 (namespace=("profile", user_id), key="health") 存 → 跨会话、跨重启

生产环境换成 PostgresSaver / PostgresStore：接口相同，graph.compile(checkpointer=…, store=…) 处换实现即可，Agent 代码不用改。
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from bluemed.config import MEMORY_DB

# checkpoint 反序列化白名单：只允许还原我们自己的这几个类型（防止从存储中还原出任意对象）
SERDE = JsonPlusSerializer(allowed_msgpack_modules=[("bluemed.schemas", n) for n in ("Evidence", "Answer", "Review")])


def open_memory(db_path: str | Path = MEMORY_DB) -> tuple[SqliteSaver, SqliteStore]:
    """打开（不存在则创建）记忆数据库，返回 (checkpointer, store)。"""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Flask 多线程处理请求 → check_same_thread=False；两个类内部都用锁串行化对连接的访问
    saver_conn = sqlite3.connect(db_path, check_same_thread=False)
    saver_conn.execute("PRAGMA journal_mode=WAL")             # 读写不互相阻塞；两个连接写同一文件时靠 busy timeout 等待
    store_conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)  # store 自己管理事务
    checkpointer = SqliteSaver(saver_conn, serde=SERDE)
    store = SqliteStore(store_conn)
    checkpointer.setup()                                       # 建表（已存在则跳过）
    store.setup()
    return checkpointer, store


def close_memory(checkpointer: SqliteSaver, store: SqliteStore) -> None:
    checkpointer.conn.close()
    store.conn.close()
