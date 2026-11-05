"""A 组：用 DeepSeek 的轨迹做 SFT（LoRA）。RL_PLAN.md 第 6.1 节。

  python -m rl.train_sft --model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 --out /root/autodl-tmp/ckpt/sft
  python -m rl.train_sft --check   # 只渲染、统计遮罩，不训练（检查模板）

loss 只算模型自己写的部分：每条 assistant 消息（工具调用 / 最后的 JSON）和它结尾的 <|im_end|>；
系统提示、工具定义、问题、工具返回全部遮掉（labels = -100）。
"""

import argparse
import json
import random
import re

from rl.make_questions import OUT_DIR

ASSISTANT = "<|im_start|>assistant\n"
END = "<|im_end|>"
# Qwen3 混合思考版模板会在最后一条 assistant 前插空 think（RL_PLAN 第 6.1 节）；推理时的生成提示里没有，训练前去掉
EMPTY_THINK = re.compile(r"(<\|im_start\|>assistant\n)<think>\n\n</think>\n\n")
TWO_HOP_REPEAT = 2                  # 两跳题只有 249 条（留下率 52%），重复一遍


def render(tok, ex: dict) -> str:
    text = tok.apply_chat_template(ex["messages"], tools=ex["tools"], tokenize=False)
    return EMPTY_THINK.sub(r"\1", text)


def encode(tok, ex: dict, max_len: int) -> dict:
    """token 化，并按字符位置把 assistant 段标成要学的部分。"""
    text = render(tok, ex)
    spans, i = [], 0
    while (s := text.find(ASSISTANT, i)) != -1:
        start = s + len(ASSISTANT)
        end = text.index(END, start) + len(END)
        spans.append((start, end))
        i = end
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    labels = [t if any(a <= lo and hi <= b for a, b in spans) else -100
              for t, (lo, hi) in zip(enc["input_ids"], enc["offset_mapping"])]
    return {"input_ids": enc["input_ids"][:max_len], "labels": labels[:max_len],
            "n_spans": len(spans), "n_learn": sum(l != -100 for l in labels)}


def load(path, two_hop_repeat: int = TWO_HOP_REPEAT) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    return rows + [r for r in rows if r["type"] == "two_hop"] * (two_hop_repeat - 1)


def check(tok) -> None:
    """模板检查：生成提示和训练文本对得上；每条 assistant 都被标成要学；没有残留的 think。"""
    rows = load(OUT_DIR / "sft.jsonl", 1)
    raw = tok.apply_chat_template(rows[0]["messages"], tools=rows[0]["tools"], tokenize=False)
    print("模板会插空 think：", bool(EMPTY_THINK.search(raw)))
    ex = rows[0]
    gen = tok.apply_chat_template(ex["messages"][:2], tools=ex["tools"], add_generation_prompt=True, tokenize=False)
    assert render(tok, ex).startswith(gen), "训练文本的前缀和推理时的生成提示不一致"
    bad = 0
    for r in rows:
        e = encode(tok, r, 10 ** 6)
        n_assist = sum(m["role"] == "assistant" for m in r["messages"])
        bad += e["n_spans"] != n_assist or "<think>" in render(tok, r)
    e = encode(tok, ex, 10 ** 6)
    learn = tok.decode([t for t, l in zip(e["input_ids"], e["labels"]) if l != -100])
    print(f"{len(rows)} 条，assistant 段数不对或残留 think 的 {bad} 条")
    print(f"第 1 条：{len(e['input_ids'])} token，要学的 {e['n_learn']} 个：\n{learn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", default="/root/autodl-tmp/ckpt/sft")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4, help="每卡 batch；有效 batch = batch × accum")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条（冒烟训练）")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.check:
        return check(tok)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    rows = load(OUT_DIR / "sft.jsonl")[:args.limit]
    random.Random(0).shuffle(rows)
    data = Dataset.from_list([{k: v for k, v in encode(tok, r, args.max_len).items() if k in ("input_ids", "labels")}
                              for r in rows])
    print(f"训练 {len(data)} 条（两跳重复 {TWO_HOP_REPEAT} 次），平均 {sum(map(len, data['input_ids'])) / len(data):.0f} token")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
            warmup_ratio=0.03, per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
            bf16=True, logging_steps=5, save_strategy="epoch", save_total_limit=2, report_to=["tensorboard"],
            dataloader_num_workers=2, seed=0),
        train_dataset=data,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    model.save_pretrained(f"{args.out}/final")
    tok.save_pretrained(f"{args.out}/final")
    print(f"LoRA 存到 {args.out}/final")


if __name__ == "__main__":
    main()
