"""SFT 数据：在 train 的一部分题上让 DeepSeek 真实做一遍，按奖励筛出好轨迹（RL_PLAN.md 第 1.2.1、3.4、6.1 节）。

  python -m rl.sft_data questions          # ① 选题：train 里按 id 固定抽约 1/4，有口语版的用口语版 → questions_sft.jsonl
  python -m rl.sft_data collect [--limit N] # ② DeepSeek 做题：先 temperature 0 做一遍，没过的用 temperature 1 再做一次
  python -m rl.sft_data export             # ③ 拒绝采样：每题留一条好轨迹 → data/rl/sft.jsonl

选中的题不进 GRPO（is_sft），所以 A / B / C 三组的 RL 题都是剩下那 3/4。
轨迹里只有工具调用和最后的 JSON：DeepSeek 的思考内容没有存，学生（Qwen3-4B-Instruct）本来也不输出思考。
"""

import argparse
import json
import random
import statistics
from collections import Counter
from types import SimpleNamespace

from rl.make_questions import OUT_DIR, TYPES
from rl.reward import parse_final
from rl.rollout import TOOLS

SFT_FRAC = 0.25                     # 8,000 道 train 题里约 2,000 道给 SFT
KEEP = 0.8                          # 奖励 ≥ 0.8 才留（和评测的"做对"同一条线）
ROUNDS = [0.0, 1.0]                 # 第一遍贪心；没过的再用 temperature 1 做一次
STUDENT = "mlx-community/Qwen3-4B-Instruct-2507-4bit"   # 只用它的分词器量长度
MAX_TOKENS = 6144                   # 第 6.1 节的最长序列


def is_sft(qid: str) -> bool:
    """按模板题 id 定，口语版和模板版归同一边。"""
    return random.Random(f"sft|{qid.removesuffix('-oral')}").random() < SFT_FRAC


def build_questions() -> list[dict]:
    train = [json.loads(l) for l in open(OUT_DIR / "questions_train.jsonl")]
    oral_path = OUT_DIR / "questions_train_oral.jsonl"
    if not oral_path.exists():
        raise SystemExit("先跑 python -m rl.rewrite --split train --frac 0.5")
    oral = {(q := json.loads(l))["id"].removesuffix("-oral"): q for l in open(oral_path)}
    qs = [oral.get(q["id"], q) for q in train if is_sft(q["id"])]
    with open(OUT_DIR / "questions_sft.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    n_oral = sum(q["id"].endswith("-oral") for q in qs)
    print(f"SFT 题 {len(qs)} 道（口语 {n_oral}，模板 {len(qs) - n_oral}）：{dict(Counter(q['type'] for q in qs))}")
    return qs


def _args(temperature: float, workers: int) -> SimpleNamespace:
    return SimpleNamespace(model="deepseek", split="sft", per_type=None, samples=1, temperature=temperature,
                           workers=workers)


def collect(limit: int | None, workers: int, rounds: int) -> None:
    from rl.eval import run
    qs = [json.loads(l) for l in open(OUT_DIR / "questions_sft.jsonl")][:limit]
    for i, t in enumerate(ROUNDS[:rounds]):
        if i:                                                   # 只重做上一轮做了但没过的题（接口出错没做的不算）
            rows = [json.loads(l) for l in open(out)]
            failed = {r["qid"] for r in rows} - {r["qid"] for r in rows if good(r)}
            qs = [q for q in qs if q["id"] in failed]
        print(f"第 {i + 1} 轮（temperature {t}）：{len(qs)} 题")
        out = run(_args(t, workers), qs)


def good(row: dict) -> bool:
    """能当 SFT 标准答案的轨迹：分数够、没有非法调用、最后一条就是一行干净的 JSON。"""
    r, final = row["reward"], (row["final"] or "").strip()
    if r["total"] < KEEP or r["invalid"] or "\n" in final:
        return False
    try:
        return parse_final(final) is not None and isinstance(json.loads(final), dict)
    except json.JSONDecodeError:
        return False


def export() -> None:
    from transformers import AutoTokenizer
    from rl.eval import EVAL_DIR
    tok = AutoTokenizer.from_pretrained(STUDENT)
    qs = [json.loads(l) for l in open(OUT_DIR / "questions_sft.jsonl")]
    rows = {}
    for t in ROUNDS:                                            # 前一轮过了的优先
        p = EVAL_DIR / f"deepseek_sft_t{t}_k1.jsonl"
        for r in map(json.loads, open(p)) if p.exists() else []:
            if r["qid"] not in rows and good(r):
                rows[r["qid"]] = r | {"round": t}
    kept, too_long, lengths = [], 0, []
    for q in qs:
        r = rows.get(q["id"])
        if not r:
            continue
        n = len(tok(tok.apply_chat_template(r["messages"], tools=TOOLS, tokenize=False))["input_ids"])
        if n > MAX_TOKENS:
            too_long += 1
            continue
        lengths.append(n)
        kept.append({"id": q["id"], "type": q["type"], "subtype": q["subtype"], "oral": q["id"].endswith("-oral"),
                     "reward": r["reward"]["total"], "round": r["round"], "tokens": n,
                     "messages": r["messages"], "tools": TOOLS})
    with open(OUT_DIR / "sft.jsonl", "w") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    print(f"留下 {len(kept)} / {len(qs)} 题（超长丢掉 {too_long}）→ {OUT_DIR / 'sft.jsonl'}")
    print("题型".ljust(14) + "题数    留下    留下率   第2轮补回   口语")
    for t in TYPES:
        n = sum(q["type"] == t for q in qs)
        k = [x for x in kept if x["type"] == t]
        print(f"{t:14s}{n:5d}{len(k):8d}{len(k) / max(1, n):9.0%}{sum(x['round'] > 0 for x in k):10d}"
              f"{sum(x['oral'] for x in k):8d}")
    if lengths:
        q = statistics.quantiles(lengths, n=20)
        print(f"token 长度：中位 {statistics.median(lengths):.0f}，p95 {q[-1]:.0f}，最长 {max(lengths)}")
        turns = Counter(sum(m["role"] == "assistant" for m in x["messages"]) for x in kept)
        print(f"assistant 轮数分布：{dict(sorted(turns.items()))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["questions", "collect", "export"])
    ap.add_argument("--limit", type=int, default=None, help="只做前 N 题（试采）")
    ap.add_argument("--rounds", type=int, default=len(ROUNDS), help="做几轮（1 = 只做贪心那一轮）")
    ap.add_argument("--workers", type=int, default=4, help="DeepSeek 并发数（余额少时上限会降，11/04 降到 6）")
    a = ap.parse_args()
    {"questions": build_questions, "collect": lambda: collect(a.limit, a.workers, a.rounds), "export": export}[a.step]()
