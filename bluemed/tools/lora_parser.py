"""调用本地 LoRA 微调的解析模型（Qwen2.5-1.5B + adapter，mlx_lm server :8080）抽取实体和关系。

大小模型协同：小模型便宜、快、专门训练过意图抽取；结果只作为给 Expert 的线索，不作为唯一依据。
服务不可用或输出不合法时返回 None，系统照常运行（优雅降级）。
"""

import json
import re

import requests

from bluemed import config
from bluemed.schemas import RELATIONS

# 必须与训练数据中的 system prompt 一字不差（见附录 A）
LORA_SYSTEM_PROMPT = (
    "你是医学问题解析器。从用户问题中抽取解决问题所需的医学实体及关系，"
    "只输出JSON：{\"entities\":[{\"name\":..,\"type\":..}],"
    "\"relations\":[{\"source\":..,\"target\":..,\"type\":..}]}"
)


def lora_parse(question: str, timeout: float = 5.0) -> dict | None:
    try:
        resp = requests.post(f"{config.PARSER_BASE_URL}/chat/completions", timeout=timeout, json={
            "model": "default_model", "temperature": 0, "max_tokens": 256,
            "messages": [{"role": "system", "content": LORA_SYSTEM_PROMPT}, {"role": "user", "content": question}]})
        text = resp.json()["choices"][0]["message"]["content"]
        obj = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        return None
    entities = [e for e in obj.get("entities", []) if isinstance(e, dict) and e.get("name")]
    relations = [r for r in obj.get("relations", []) if isinstance(r, dict) and r.get("type") in RELATIONS]
    return {"entities": entities, "relations": relations}
