"""B / C 组：多轮工具调用的 GRPO（RL_PLAN.md 第 1.6、1.8、6.2 节）。一张卡上 vLLM 采样 + HF 模型训练 LoRA。

  # 1. 筛题：用起点模型每题做 4 次，4 次全满分的题去掉（B 组起点只有约 1/5 的题组内有差异，第 7.0 节）
  python -m rl.grpo prefilter --init base --out /root/autodl-tmp/ckpt/grpo_b
  # 2. 训练
  python -m rl.grpo train --init base --out /root/autodl-tmp/ckpt/grpo_b --steps 300
  python -m rl.grpo train --init /root/autodl-tmp/ckpt/sft/final --out /root/autodl-tmp/ckpt/grpo_c --steps 120

每一步：抽题 → 每题采 G 条轨迹（vLLM，temperature 1）→ 环境执行工具、奖励函数打分 →
组内标准化得到优势，丢掉组内奖励全相同的组 → 只对模型自己生成的 token 算策略梯度 → 更新 LoRA → 存盘让 vLLM 换上。

和 rl/rollout.py（评测用）的两处不同，都是为了"训练时算概率的 token = 采样时生成的 token"：
  - 历史里的 assistant 消息存模型的原始输出（含 <tool_call> 原文），不是解析后重新拼的 JSON；
  - 调用额度用完后仍然提供工具（环境返回"已达到上限"），保证整条轨迹的系统提示一样。
严格 on-policy：每批轨迹只更新一次（同 TRL 默认 num_iterations=1），所以 ratio 恒为 1，裁剪只在数值上保留；
没有参考模型（β = 0）；loss 按 token 平均（DAPO）。
"""

import argparse
import json
import math
import random
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from rl.env import ExpertEnv
from rl.make_questions import OUT_DIR
from rl.prompt import EXPERT_PROMPT_RL
from rl.reward import score
from rl.rollout import TOOLS, parse_qwen
from rl.eval import SOLVED
from rl.sft_data import is_sft
from rl.train_sft import encode

MAX_TURNS = 8
CLIP_LOW, CLIP_HIGH = 0.2, 0.28                       # DAPO 的不对称裁剪（第 1.6 节）
MODEL = "/root/autodl-tmp/models/Qwen3-4B-Instruct-2507"


# ---------------------------------------------------------------- 题目
def rl_pool(seed: int = 0) -> list[dict]:
    """GRPO 的题：train 里没给 SFT 的那 3/4；有口语版的一半换成口语版（按 id 固定）。"""
    train = [json.loads(l) for l in open(OUT_DIR / "questions_train.jsonl")]
    oral = {(q := json.loads(l))["id"].removesuffix("-oral"): q for l in open(OUT_DIR / "questions_train_oral.jsonl")}
    pool = []
    for q in train:
        if is_sft(q["id"]):
            continue
        use_oral = q["id"] in oral and random.Random(f"grpo_oral|{q['id']}|{seed}").random() < 0.5
        pool.append(oral[q["id"]] if use_oral else q)
    return pool


# ---------------------------------------------------------------- 批量多轮采样
@dataclass
class Traj:
    q: dict
    messages: list[dict]
    env: ExpertEnv = field(default_factory=ExpertEnv)
    final: str | None = None
    turns: int = 0
    reward: object = None


def rollout_batch(qs: list[dict], generate, tok, max_turns: int = MAX_TURNS) -> list[Traj]:
    """qs 里每一项做一条轨迹。每一轮把所有还没结束的轨迹一起交给 generate(prompts) → 原始文本列表。"""
    trajs = [Traj(q, [{"role": "system", "content": EXPERT_PROMPT_RL}, {"role": "user", "content": q["question"]}])
             for q in qs]
    active = list(trajs)
    while active:
        prompts = [tok.apply_chat_template(t.messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
                   for t in active]
        texts = generate(prompts)
        still = []
        for t, text in zip(active, texts):
            t.turns += 1
            parsed = parse_qwen(text)
            t.messages.append({"role": "assistant", "content": text})       # 原始输出，训练时原样渲染
            if not parsed["tool_calls"]:
                t.final = parsed["content"]
                continue
            for c, r in zip(parsed["tool_calls"], t.env.step(parsed["tool_calls"])):
                t.messages.append({"role": "tool", "name": c["name"], "content": r.content})
            if t.turns < max_turns:
                still.append(t)
        active = still
    for t in trajs:
        t.reward = score(t.q, t.final, t.env)
    return trajs


# ---------------------------------------------------------------- 优势
def group_advantages(rewards: list[float], eps: float = 1e-4) -> list[float] | None:
    """GRPO：组内标准化。组内奖励全相同（标准差≈0）返回 None，这一组不参与训练（DAPO 动态采样）。"""
    sd = statistics.pstdev(rewards)
    if sd < eps:
        return None
    mu = statistics.mean(rewards)
    return [(r - mu) / sd for r in rewards]


# ---------------------------------------------------------------- loss
def policy_loss(model, batch: list[dict], advantages: list[float], n_tokens: int, pad_id: int):
    """batch：encode() 的输出（input_ids / labels）。只在要学的位置过 lm_head，省显存（词表 15 万）。
    返回 (loss, 统计)。loss 已除以整批要学的 token 数 n_tokens（梯度累积时各 micro-batch 直接相加）。"""
    import torch
    device = next(model.parameters()).device
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n], lab[i, :n], att[i, :n] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
    ids, lab, att = ids.to(device), lab.to(device), att.to(device)

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    target = lab[:, 1:]
    mask = target != -100
    logits = base.lm_head(hidden[mask]).float()                    # [要学的 token 数, 词表]
    logp_all = torch.log_softmax(logits, -1)
    logp = logp_all.gather(-1, target[mask].unsqueeze(-1)).squeeze(-1)
    adv = torch.tensor(advantages, device=device, dtype=torch.float32)[:, None].expand_as(target)[mask]

    ratio = torch.exp(logp - logp.detach())                        # on-policy：恒为 1，梯度 = A · ∇log π
    pg = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - CLIP_LOW, 1 + CLIP_HIGH) * adv)
    loss = pg.sum() / n_tokens
    with torch.no_grad():
        entropy = -(logp_all.exp() * logp_all).sum(-1).sum().item()
    return loss, {"tokens": int(mask.sum()), "entropy_sum": entropy, "logp_sum": logp.detach().sum().item()}


# ---------------------------------------------------------------- 服务器上的部件
class VLLMGen:
    """vLLM 采样，每次 generate 都带上当前 LoRA。"""
    def __init__(self, model: str, mem: float, max_tokens: int = 512):
        from vllm import LLM
        self.llm = LLM(model, enable_lora=True, max_lora_rank=64, max_loras=2, max_model_len=8192,
                       gpu_memory_utilization=mem, enable_prefix_caching=True, seed=0)
        self.max_tokens, self.lora, self.n = max_tokens, None, 0

    def set_lora(self, path: str) -> None:
        from vllm.lora.request import LoRARequest
        self.n += 1                                                 # 每次换新 id，vLLM 不会用旧缓存
        self.lora = LoRARequest(f"step{self.n}", self.n, path)

    def __call__(self, prompts: list[str], temperature: float = 1.0) -> list[str]:
        from vllm import SamplingParams
        # 不给单个请求设 seed：同一道题复制 G 份、同一个 seed 会采出 G 条一模一样的轨迹，组内就没有差异了
        params = SamplingParams(temperature=temperature, max_tokens=self.max_tokens)
        outs = self.llm.generate(prompts, params, lora_request=self.lora, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def load_policy(init: str, rank: int):
    """init = base：新建一个零初始化的 LoRA（等价于原版）；否则从已有 LoRA（SFT）接着训。"""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if init == "base":
        return get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                                target_modules="all-linear", task_type="CAUSAL_LM"))
    return PeftModel.from_pretrained(model, init, is_trainable=True)


def summarize(trajs: list[Traj]) -> dict:
    R = [t.reward for t in trajs]
    return {"reward": statistics.mean(r.total for r in R),
            "solved": statistics.mean(r.total >= SOLVED for r in R),
            "format_ok": statistics.mean(r.format_ok for r in R),
            "calls": statistics.mean(r.calls for r in R),
            "invalid": statistics.mean(r.invalid for r in R),
            "turns": statistics.mean(t.turns for t in trajs)}


def prefilter(args) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    gen = VLLMGen(MODEL, mem=0.85)
    if args.init != "base":
        gen.set_lora(args.init)
    pool = rl_pool()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept, stats, t0 = [], {"all_solved": 0, "kept": 0}, time.time()
    for i in range(0, len(pool), 256):
        chunk = pool[i:i + 256]
        trajs = rollout_batch([q for q in chunk for _ in range(args.prefilter_k)],
                              lambda p: gen(p, 1.0), tok)
        for j, q in enumerate(chunk):
            rs = [t.reward.total for t in trajs[j * args.prefilter_k:(j + 1) * args.prefilter_k]]
            if min(rs) >= 1.0:
                stats["all_solved"] += 1                            # 每次都满分：GRPO 学不到东西
            else:
                kept.append(q | {"prefilter": rs})
                stats["kept"] += 1
        print(f"  {i + len(chunk)}/{len(pool)} 题，保留 {stats['kept']}，全满分 {stats['all_solved']}，"
              f"{time.time() - t0:.0f}s", flush=True)
    with open(out / "pool.jsonl", "w") as f:
        for q in kept:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"筛题完成：{len(pool)} → {len(kept)} → {out / 'pool.jsonl'}")


def train(args) -> None:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool_path = out / "pool.jsonl"
    pool = [json.loads(l) for l in open(pool_path)] if pool_path.exists() else rl_pool()
    dev = [json.loads(l) for l in open(OUT_DIR / "questions_dev.jsonl")]
    dev = [q for t in ("forward", "backward", "symptoms", "two_hop", "attribute", "unanswerable")
           for q in [x for x in dev if x["type"] == t][:10]]          # 每类 10 题，贪心，看曲线
    print(f"题池 {len(pool)} 题（{'筛过' if pool_path.exists() else '没筛'}），dev 曲线用 {len(dev)} 题")

    tok = AutoTokenizer.from_pretrained(MODEL)
    gen = VLLMGen(MODEL, mem=args.vllm_mem)                        # 先起 vLLM 占住显存，再加载训练模型
    model = load_policy(args.init, args.rank)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    writer = SummaryWriter(out / "tb")
    log = open(out / "log.jsonl", "a")

    def sync(step: int) -> None:
        path = out / "adapter_live" / f"step{step}"
        model.save_pretrained(path)
        gen.set_lora(str(path))
        for old in (out / "adapter_live").glob("step*"):
            if old != path:
                shutil.rmtree(old, ignore_errors=True)

    def evaluate(step: int) -> float:
        trajs = rollout_batch(dev, lambda p: gen(p, 0.0), tok)
        s = summarize(trajs)
        for k, v in s.items():
            writer.add_scalar(f"dev/{k}", v, step)
        log.write(json.dumps({"step": step, "dev": s}, ensure_ascii=False) + "\n")
        log.flush()
        print(f"  [dev] step {step}: reward {s['reward']:+.3f} solved {s['solved']:.0%} calls {s['calls']:.2f}", flush=True)
        return s["reward"]

    rng = random.Random(args.seed)
    sync(0)
    best, stale = evaluate(0), 0
    model.save_pretrained(out / "best")
    for step in range(1, args.steps + 1):
        t0 = time.time()
        # 1. 采样：抽 oversample 倍的题，丢掉组内奖励全相同的组，凑够 n_groups 组
        groups, all_trajs, tries = [], [], 0
        while len(groups) < args.n_groups and tries < args.max_tries:
            tries += 1
            qs = rng.sample(pool, args.n_groups * args.oversample)
            trajs = rollout_batch([q for q in qs for _ in range(args.g)], lambda p: gen(p, 1.0), tok)
            all_trajs += trajs
            for j in range(len(qs)):
                grp = trajs[j * args.g:(j + 1) * args.g]
                adv = group_advantages([t.reward.total for t in grp])
                if adv is not None and len(groups) < args.n_groups:
                    groups.append((grp, adv))
        t_roll = time.time() - t0
        if not groups:
            print(f"step {step}: 没有组内有差异的组，跳过", flush=True)
            continue

        # 2. 训练：只对模型自己生成的 token 算 loss，按 token 平均
        items = [(encode(tok, {"messages": t.messages, "tools": TOOLS}, args.max_len), a)
                 for grp, adv in groups for t, a in zip(grp, adv)]
        n_tokens = sum(sum(l != -100 for l in e["labels"]) for e, _ in items)
        model.train()
        opt.zero_grad(set_to_none=True)
        tok_sum = ent_sum = 0.0
        for i in range(0, len(items), args.micro):
            mb = items[i:i + args.micro]
            loss, st = policy_loss(model, [e for e, _ in mb], [a for _, a in mb], n_tokens, tok.pad_token_id)
            loss.backward()
            tok_sum += st["tokens"]
            ent_sum += st["entropy_sum"]
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sync(step)

        s = summarize(all_trajs)
        rec = {"step": step, **s, "groups": len(groups), "sampled_groups": len(all_trajs) // args.g,
               "varied": len(groups) / (len(all_trajs) // args.g), "entropy": ent_sum / max(1, tok_sum),
               "grad_norm": gnorm, "learn_tokens": n_tokens, "t_rollout": t_roll, "t_step": time.time() - t0}
        for k, v in rec.items():
            if k != "step" and isinstance(v, (int, float)) and not math.isnan(v):
                writer.add_scalar(f"train/{k}", v, step)
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print(f"step {step}: reward {s['reward']:+.3f} solved {s['solved']:.0%} 有差异 {rec['varied']:.0%} "
              f"熵 {rec['entropy']:.3f} 梯度 {gnorm:.2f} 调用 {s['calls']:.2f} | 采样 {t_roll:.0f}s 共 {rec['t_step']:.0f}s",
              flush=True)
        if step % 10 == 0:                                         # 存几条轨迹，看模型在学什么、有没有钻空子
            with open(out / "samples.jsonl", "a") as f:
                for t in rng.sample(all_trajs, 4):
                    f.write(json.dumps({"step": step, "q": t.q["question"], "reward": t.reward.to_dict(),
                                        "messages": t.messages[1:]}, ensure_ascii=False) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained(out / f"step{step}")
        if step % args.eval_every == 0:
            # 提前停：dev 连续 patience 次没比最好成绩高 min_delta 就停（dev 只有 60 题，波动约 ±0.03）
            r = evaluate(step)
            if r > best + args.min_delta:
                best, stale = r, 0
                model.save_pretrained(out / "best")
                (out / "best" / "step.txt").write_text(str(step))
            else:
                stale += 1
            if step >= args.min_steps and stale >= args.patience:
                print(f"dev 连续 {stale} 次没有提升（最好 {best:+.3f}），第 {step} 步提前停止", flush=True)
                break
    model.save_pretrained(out / "final")
    print(f"训练完成 → {out / 'final'}；dev 最好的在 {out / 'best'}（{best:+.3f}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prefilter", "train"])
    ap.add_argument("--init", default="base", help="base = 原版（B 组）；或 SFT 的 LoRA 目录（C 组）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-groups", type=int, default=16, help="每步用多少道题（组）")
    ap.add_argument("--g", type=int, default=8, help="每题采几条轨迹")
    ap.add_argument("--oversample", type=int, default=2, help="每次抽 n_groups 的几倍题，丢掉组内无差异的")
    ap.add_argument("--max-tries", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--micro", type=int, default=4, help="micro-batch 条数（梯度累积）")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--min-steps", type=int, default=150, help="至少训这么多步才允许提前停")
    ap.add_argument("--patience", type=int, default=3, help="dev 连续几次没提升就停")
    ap.add_argument("--min-delta", type=float, default=0.01)
    ap.add_argument("--prefilter-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"prefilter": prefilter, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()
