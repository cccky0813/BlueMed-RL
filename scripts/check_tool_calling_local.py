"""重测本地模型的 tool calling（11/02）：Phase 0 的 0/3 是模板 + 解析器的问题，不是模型不会调用。

Qwen2.5 官方 chat template 里的调用示例写成 {{"name": ..., "arguments": ...}}（双花括号），
小模型照抄后输出不是合法 JSON，mlx_lm server 解析失败就把调用丢掉（只打一条 warning）。

这里用真实的 EXPERT_PROMPT + 5 个图谱工具，只看第一步调用（不执行工具），三种设置：
  原样     原模板 + 严格 json.loads（= mlx_lm server 的行为）
  宽容解析 同一份输出，去掉多余的一层花括号再解析
  修模板   把模板里的示例改成单花括号后重新生成，严格解析
对照：DeepSeek（线上 Expert 用的模型，原生 tool calling）。

用法：
  python scripts/check_tool_calling_local.py                      # 本地 3B + 1.5B
  python scripts/check_tool_calling_local.py --deepseek           # 再加 DeepSeek 对照
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from bluemed.agents.expert_agent import EXPERT_PROMPT
from bluemed.tools.kg_tools import KG_TOOLS

TOOLS = [convert_to_openai_tool(t) for t in KG_TOOLS]
# 模板源码里示例在 Jinja 字符串内，引号是转义过的
BAD_EXAMPLE = r'{{\"name\": <function-name>, \"arguments\": <args-json-object>}}'
GOOD_EXAMPLE = r'{\"name\": <function-name>, \"arguments\": <args-json-object>}'


def kg(entity, *relations, direction="forward"):
    """期望 kg_search：实体名包含 entity，关系在 relations 里，方向一致。"""
    return lambda name, a: (name == "kg_search" and entity in a.get("entity", "")
                            and a.get("relation") in relations and a.get("direction", "forward") == direction)


def info(disease, field):
    return lambda name, a: name == "get_disease_info" and disease in a.get("disease", "") and field in a.get("fields", [])


def symptoms(*names):
    return lambda name, a: name == "find_diseases_by_symptoms" and all(any(n in s for s in a.get("symptoms", [])) for n in names)


def multi_hop(entity):
    return lambda name, a: name == "expand_multi_hop" and entity in a.get("entity", "")


def either(*checks):
    return lambda name, a: any(c(name, a) for c in checks)


# (问题, 判定函数)：多种合理做法都算对
CASES = [
    ("我爸有痛风，平时不能吃什么？", kg("痛风", "SHOULD_NOT_EAT")),
    ("糖尿病要做哪些检查？", kg("糖尿病", "REQUIRES_CHECK")),
    ("高血压一般吃什么药？", kg("高血压", "RECOMMENDS_DRUG", "COMMONLY_USES_DRUG")),
    ("肺炎应该挂什么科？", kg("肺炎", "TREATED_BY", "BELONGS_TO")),     # 图里两种关系的数据完全相同（8806/8806）
    ("感冒有哪些症状？", kg("感冒", "HAS_SYMPTOM")),
    ("胃炎平时适合吃什么？", kg("胃炎", "SHOULD_EAT", "RECOMMENDS_RECIPE")),
    ("哮喘会有哪些并发症？", kg("哮喘", "ACCOMPANIES")),
    ("小孩中耳炎不能吃什么？", either(kg("中耳炎", "SHOULD_NOT_EAT"),
                                     lambda n, a: n == "link_entity" and "中耳炎" in a.get("name", ""))),
    ("头痛可能是什么病？", either(kg("头痛", "HAS_SYMPTOM", direction="backward"), symptoms("头痛"))),
    ("头痛又发烧，可能是什么病？", symptoms("头痛", "发")),
    ("最近咳嗽、胸痛还发热，是什么病？", symptoms("咳嗽", "胸痛")),
    ("糖尿病是什么原因引起的？", info("糖尿病", "cause")),
    ("怎么预防骨质疏松？", info("骨质疏松", "prevent")),
    ("白血病的治愈率高吗？", info("白血病", "cured_prob")),
    ("糖尿病的并发症一般要做哪些检查？", either(multi_hop("糖尿病"), kg("糖尿病", "ACCOMPANIES"))),
]


def parse_calls(text: str, lenient: bool) -> tuple[bool, list[dict]]:
    """返回（是否发出了 <tool_call>，解析成功的调用列表）。"""
    blocks = re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S)
    calls = []
    for b in blocks:
        b = b.strip()
        candidates = [b]
        if lenient and b.startswith("{{"):
            candidates.append(b[1:-1] if b.endswith("}}}") else b[1:])
        for c in candidates:
            try:
                obj = json.loads(c)
                calls.append({"name": obj["name"], "args": obj.get("arguments") or {}})
                break
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return bool(blocks), calls


def run_local(model_id: str) -> dict:
    from mlx_lm import generate, load
    model, tok = load(model_id)
    fixed_template = tok.chat_template.replace(BAD_EXAMPLE, GOOD_EXAMPLE)
    if fixed_template == tok.chat_template:
        print(f"  {model_id}：模板里没有双花括号示例（Qwen3 起已修复），'修模板' 和 '原样' 相同")

    def gen(question, template):
        messages = [{"role": "system", "content": EXPERT_PROMPT}, {"role": "user", "content": question}]
        prompt = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False,
                                         chat_template=template)
        return generate(model, tok, prompt=prompt, max_tokens=300, verbose=False)

    rows = []
    for question, check in CASES:
        t0 = time.perf_counter()
        raw = gen(question, tok.chat_template)
        fixed = gen(question, fixed_template)
        latency = (time.perf_counter() - t0) / 2
        row = {"question": question, "latency": latency, "raw": raw, "fixed_raw": fixed}
        for setting, text, lenient in [("原样", raw, False), ("宽容解析", raw, True), ("修模板", fixed, False)]:
            emitted, calls = parse_calls(text, lenient)
            row[setting] = {"emitted": emitted, "parsed": bool(calls),
                            "correct": any(check(c["name"], c["args"]) for c in calls), "calls": calls}
        rows.append(row)
        print(f"  {model_id.split('/')[-1]} | {question} | "
              + " ".join(f"{s}:{'✅' if row[s]['correct'] else ('⚠️' if row[s]['parsed'] else '❌')}"
                         for s in ["原样", "宽容解析", "修模板"]), flush=True)
    return {"model": model_id, "rows": rows}


def run_deepseek() -> dict:
    from bluemed.llm import get_llm
    llm = get_llm().bind_tools(KG_TOOLS)            # 和线上 Expert 一样：保留思考
    rows = []
    for question, check in CASES:
        t0 = time.perf_counter()
        msg = llm.invoke([SystemMessage(EXPERT_PROMPT), HumanMessage(question)])
        calls = [{"name": c["name"], "args": c["args"]} for c in msg.tool_calls]
        ok = any(check(c["name"], c["args"]) for c in calls)
        rows.append({"question": question, "latency": time.perf_counter() - t0,
                     "原生": {"emitted": bool(calls), "parsed": bool(calls), "correct": ok, "calls": calls}})
        print(f"  deepseek | {question} | {'✅' if ok else '❌'} {calls}", flush=True)
    return {"model": "deepseek (" + get_llm().model_name + ")", "rows": rows}


def summarize(result: dict) -> None:
    rows = result["rows"]
    settings = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    print(f"\n== {result['model']}（{len(rows)} 题，平均 {sum(r['latency'] for r in rows) / len(rows):.1f}s/题）")
    for s in settings:
        n = lambda key: sum(r[s][key] for r in rows)
        print(f"   {s:6s} 发出调用 {n('emitted'):2d}  可解析 {n('parsed'):2d}  选对 {n('correct'):2d} / {len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["mlx-community/Qwen2.5-3B-Instruct-4bit",
                                                    "mlx-community/Qwen2.5-1.5B-Instruct-4bit"])
    ap.add_argument("--deepseek", action="store_true")
    ap.add_argument("--out", default="outputs/tool_calling_local.json")
    args = ap.parse_args()

    results = [run_local(m) for m in args.models] + ([run_deepseek()] if args.deepseek else [])
    for r in results:
        summarize(r)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\n逐题原始输出已保存到 {args.out}")
