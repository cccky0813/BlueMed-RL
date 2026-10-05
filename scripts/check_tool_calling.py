"""比较几个候选模型的 tool calling 能力：是否调用了正确的工具、参数是否正确、耗时多少。"""

import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.tools import tool

from bluemed.llm import get_llm


@tool
def kg_search(entity: str,
              relation: Literal["HAS_SYMPTOM", "SHOULD_EAT", "SHOULD_NOT_EAT", "RECOMMENDS_DRUG", "TREATED_BY"],
              direction: Literal["forward", "backward"] = "forward") -> str:
    """在医疗知识图谱中按关系查询。forward：已知疾病查目标（如痛风不能吃什么）；backward：已知目标反查疾病（如头痛是什么病）。"""
    return "（测试用，不真正查询）"


@tool
def get_disease_info(disease: str, field: Literal["cause", "prevent", "desc"]) -> str:
    """查询疾病的文字属性：cause 病因、prevent 预防、desc 简介。"""
    return "（测试用，不真正查询）"


CASES = [
    ("我爸有痛风，平时不能吃什么？", "kg_search", {"entity": "痛风", "relation": "SHOULD_NOT_EAT"}),
    ("头痛可能是什么病？", "kg_search", {"entity": "头痛", "relation": "HAS_SYMPTOM", "direction": "backward"}),
    ("糖尿病是什么原因引起的？", "get_disease_info", {"disease": "糖尿病", "field": "cause"}),
]

MODELS = sys.argv[1:] or ["gpt-4o-mini", "gpt-4.1-mini", "deepseek-chat", "qwen-plus"]

for model in MODELS:
    llm = get_llm(model).bind_tools([kg_search, get_disease_info])
    correct, total_time = 0, 0.0
    for question, want_tool, want_args in CASES:
        start = time.perf_counter()
        try:
            msg = llm.invoke(question)
        except Exception as e:
            print(f"  {model}: 调用失败 {str(e)[:80]}")
            break
        total_time += time.perf_counter() - start
        calls = msg.tool_calls
        ok = bool(calls) and calls[0]["name"] == want_tool and all(calls[0]["args"].get(k) == v for k, v in want_args.items())
        correct += ok
        got = f'{calls[0]["name"]}({calls[0]["args"]})' if calls else f"没有调用工具：{msg.content[:40]}"
        print(f"  {'✅' if ok else '❌'} {model} | {question} → {got}")
    else:
        print(f"== {model}: {correct}/{len(CASES)} 正确，平均 {total_time / len(CASES):.2f}s\n")
