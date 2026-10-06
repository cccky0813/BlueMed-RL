"""读取 .env 里的配置，整个项目只在这里读环境变量。"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")

PARSER_BASE_URL = os.getenv("PARSER_BASE_URL", "http://localhost:8080/v1")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "123456789")

# 记忆数据库（短期记忆 checkpoint + 长期记忆档案；data/ 已 gitignore）
MEMORY_DB = os.getenv("MEMORY_DB", str(ROOT / "data" / "memory.db"))
