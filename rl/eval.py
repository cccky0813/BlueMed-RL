"""批量评测：让一个模型把一组题完整做一遍，存下每条轨迹，按题型汇总分数（RL_PLAN.md 第 7 节）。

用法：
  python -m rl.eval --model mlx-community/Qwen3-4B-Instruct-2507-4bit            # 本地 4B，dev 全部，贪心解码
  python -m rl.eval --model deepseek                                            # DeepSeek（线上 Expert 的模型）
  python -m rl.eval --model <id> --per-type 10 --samples 8 --temperature 1.0    # 每类 10 题，每题 8 次 → pass@8
  python -m rl.eval --model <本地路径> --engine vllm [--lora <adapter 目录>]   # 服务器上用 vLLM
  python -m rl.eval --summary outputs/eval/xxx.jsonl                            # 只汇总已有结果

结果写到 outputs/eval/<模型>_<split>_t<温度>_k<次数>.jsonl，一行一条轨迹；中断后重跑会跳过做过的。
"""

import argparse
import json
import statistics
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rl.make_questions import OUT_DIR, TYPES
from rl.rollout import DeepSeekPolicy, MLXPolicy, VLLMPolicy, rollout

EVAL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "eval"
SOLVED = 0.8                        # 总分 ≥ 0.8 算"做对"（pass@k 用）


def load_questions(split: str, per_type: int | None) -> list[dict]:
    qs = [json.loads(l) for l in open(OUT_DIR / f"questions_{split}.jsonl")]
    if per_type:
        qs = [q for t in TYPES for q in [q for q in qs if q["type"] == t][:per_type]]
    return qs


def _serialize_embedding() -> None:
    """多线程时 m3e 会被几个线程同时加载、同时推理，进程直接崩掉（11/03 实测跑到第 8 题退出）。
    先在主线程加载好，推理加一把锁（只有 expand_multi_hop / 实体链接会用到，开销很小）。"""
    import bluemed.embedding as emb
    emb._model()
    lock, embed = threading.Lock(), emb.embed

    def locked(texts):
        with lock:
            return embed(texts)
    emb.embed = locked


def run(args, qs: list[dict] | None = None) -> Path:
    """qs：只做这些题（rl/sft_data.py 重做没过的题时用），默认按 --split / --per-type 读题。"""
    qs = load_questions(args.split, args.per_type) if qs is None else qs
    name = "deepseek" if args.model == "deepseek" else args.model.rstrip("/").split("/")[-1]
    if getattr(args, "lora", None):                  # 取最后两级（grpo_b/best → grpo_b-best），否则 B、C 的 best 会撞名
        name += "+" + "-".join(args.lora.rstrip("/").split("/")[-2:])
    out = EVAL_DIR / f"{name}_{args.split}_t{args.temperature}_k{args.samples}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["qid"], r["sample"]) for r in map(json.loads, open(out))} if out.exists() else set()
    todo = [(q, k) for q in qs for k in range(args.samples) if (q["id"], k) not in done]
    print(f"{name}：{len(qs)} 题 × {args.samples} 次，已有 {len(done)}，待做 {len(todo)} → {out}")

    def work(item, policy):
        q, k = item
        t = rollout(q, policy)
        return {"sample": k, "type": q["type"], "subtype": q["subtype"], "question": q["question"],
                "gold": q["gold"], "min_calls": q["min_calls"], **t.to_dict()}

    with open(out, "a") as f:
        def save(i, row):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            r = row["reward"]
            print(f"  [{i + 1}/{len(todo)}] {row['type']:12s} {r['total']:+.2f} 调用 {r['calls']} "
                  f"{row['seconds']:.1f}s  {row['question'][:30]}", flush=True)

        if args.model == "deepseek":                       # API：多线程，每个线程各自一个客户端
            _serialize_embedding()

            def safe(item):                                # 限流、超时等接口错误：这题先不存，重跑时会再做
                try:
                    return work(item, DeepSeekPolicy(args.temperature))
                except Exception as e:
                    print(f"  接口错误，跳过 {item[0]['id']}：{type(e).__name__} {str(e)[:120]}", flush=True)

            errors = 0
            with ThreadPoolExecutor(args.workers) as pool:
                for i, row in enumerate(pool.map(safe, todo)):
                    if row:
                        save(i, row)
                    else:
                        errors += 1
            if errors:
                print(f"{errors} 题接口出错没做，重跑同一命令会补上")
        else:
            policy = (VLLMPolicy(args.model, temperature=args.temperature, lora=args.lora)
                      if getattr(args, "engine", "mlx") == "vllm" else MLXPolicy(args.model, temperature=args.temperature))
            for i, item in enumerate(todo):
                save(i, work(item, policy))
    return out


def summarize(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path)]
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    k = max(len(v) for v in by_q.values())

    def stats(rs: list[dict]) -> dict:
        rw = [r["reward"] for r in rs]
        answerable = [x for r, x in zip(rs, rw) if "insufficient" not in r["gold"]]
        qids = {r["qid"] for r in rs}
        return {
            "n": len(rs),
            "reward": statistics.mean(x["total"] for x in rw),
            "solved": statistics.mean(x["total"] >= SOLVED for x in rw),
            "format_ok": statistics.mean(x["format_ok"] for x in rw),
            "abstain_on_answerable": statistics.mean(x["status"] == "insufficient" for x in answerable) if answerable else None,
            "ungrounded": sum(x["n_ungrounded"] for x in rw) / max(1, sum(x["n_answer"] for x in rw)),
            "calls": statistics.mean(x["calls"] for x in rw),
            "invalid": statistics.mean(x["invalid"] for x in rw),
            "seconds": statistics.mean(r["seconds"] for r in rs),
            f"pass@{k}": statistics.mean(any(r["reward"]["total"] >= SOLVED for r in by_q[q]) for q in qids),
            "group_varies": statistics.mean(len({r["reward"]["total"] for r in by_q[q]}) > 1 for q in qids),
        }

    result = {"file": path.name, "k": k, "all": stats(rows),
              "by_type": {t: stats([r for r in rows if r["type"] == t]) for t in TYPES if any(r["type"] == t for r in rows)}}
    print_summary(result)
    return result


def print_summary(res: dict) -> None:
    k = res["k"]
    cols = [("reward", "平均分", "+.2f"), ("solved", "做对率", ".0%"), ("format_ok", "格式对", ".0%"),
            ("abstain_on_answerable", "能答却弃答", ".0%"), ("ungrounded", "无据答案", ".0%"),
            ("calls", "调用", ".1f"), ("invalid", "非法", ".2f"), ("seconds", "秒/题", ".1f")]
    if k > 1:
        cols += [(f"pass@{k}", f"pass@{k}", ".0%"), ("group_varies", "组内有差异", ".0%")]
    print(f"\n== {res['file']}")
    print("题型".ljust(13) + "".join(c[1].rjust(9) for c in cols) + "   题数")
    for name, s in [*res["by_type"].items(), ("全部", res["all"])]:
        cells = ["—" if s[key] is None else format(s[key], fmt) for key, _, fmt in cols]
        print(name.ljust(13) + "".join(c.rjust(9) for c in cells) + f"   {s['n']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--per-type", type=int, default=None, help="每类只取前 N 题")
    ap.add_argument("--samples", type=int, default=1, help="每题做几次（>1 时算 pass@k）")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8, help="DeepSeek 并发数")
    ap.add_argument("--engine", choices=["mlx", "vllm"], default="mlx", help="本地模型用 mlx（Mac）还是 vLLM（服务器）")
    ap.add_argument("--lora", default=None, help="vLLM 时挂的 LoRA 目录（SFT / GRPO 的输出）")
    ap.add_argument("--summary", default=None, help="只汇总这个结果文件")
    args = ap.parse_args()
    if args.summary:
        summarize(Path(args.summary))
    else:
        summarize(run(args))
