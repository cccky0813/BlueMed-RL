"""口语改写：用 DeepSeek 把模板题改写成真实用户的问法（RL_PLAN.md 第 3.3 节第 3、4 步）。

只换问法，不换题：id 加 "-oral"，标准答案、oracle、题型原样保留，原题存在 template 字段。
改写结果写到 data/rl/questions_<split>_oral.jsonl，评测时用 --split <split>_oral。

校验（第 4 步）：让 DeepSeek 同时说出"原实体在改写句里写成了什么"，程序检查
  ① 这个说法确实出现在改写句里；② 用线上的 _link 能链接回原实体。
不通过的把原因告诉 DeepSeek 重写一次，还不通过就丢掉。这样保证题目仍然能靠工具查到答案。

用法：
  python -m rl.rewrite --split dev               # dev 全部改写（和模板题一一对照）
  python -m rl.rewrite --split train --frac 0.5  # train 改写一半
中断后重跑会跳过已改写的题。
"""

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from rl.kg_memory import use_memory_graph
from rl.make_questions import BACKWARD, OUT_DIR

ATTEMPTS = 2
STYLES = [
    "患者本人在手机上随口问，句子短，不太讲究语法",
    "家属替病人问，先交代一句是谁、什么情况，再问问题",
    "带一点语气词（啊、呢、吧、咋），但不要夸张",
    "先简单描述自己的经历或顾虑，问题放在最后",
    "像在搜索框里输入，只有几个关键词，没有完整句子",
    "说话比较客气，像在问医生",
    "用大白话换一种问法（比如\"挂什么科\"说成\"该去看哪个科\"，\"忌口\"说成\"哪些东西不能碰\"）",
]
# 第一版（11/04）"急死人了""病友们"这类套话在 300 题里出现了约 60 次，还有 3 题把题意说明抄进了问题，这里直接禁掉
BANNED = ["急死", "病友", "群里", "本身"]

PROMPT = """把下面这道医疗问题改写成真实用户会问的口语问法。

原题：{question}
这道题问的是：{intent}
改写风格：{style}

要求：
1. 问的内容必须完全不变：还是问同一件事（{intent}），不能多问、少问或换成别的问题；是非题（能不能吃）改写后还是是非题；
2. 下面这些实体必须保留，让人一看就知道指的是哪个：{entities}
   可以加口语修饰（"我家老人的糖尿病""那个叫 xx 的病"），可以用常见简称，但不能换成另一种东西，也不能写错字；
3. 不要加入原题没有的新症状、新疾病、新食物，不要给出任何答案或医学建议；
4. 一句或两句话，不超过 60 个字；不要用"急死人了""病友们"之类的套话；
5. "这道题问的是"只是告诉你题意，不要把里面的括号说明写进问题；
6. 只问一个问题：不要拆成两问（"会有哪些并发症？这些并发症有啥症状？"不行），也不要顺带多问（问食谱时别再问"吃啥好"）。

mentions 里逐个写出上面每个实体在你的改写句里是怎么写的（必须是改写句里原样出现的片段）。"""


class Mention(BaseModel):
    original: str = Field(description="原题里的实体名")
    said: str = Field(description="这个实体在改写句里的写法，必须是改写句的原文片段")


class Rewrite(BaseModel):
    question: str = Field(description="改写后的问题")
    mentions: list[Mention]


def key_entities(q: dict) -> list[tuple[str, str]]:
    """改写后必须能找回来的实体：(名称, 节点类型)。"""
    if q["type"] == "symptoms":
        return [(s, "Symptom") for s in q["symptoms"]]
    if q["type"] == "backward":
        return [(q["anchor"], BACKWARD[q["subtype"]][1])]
    ents = [(q["anchor"], "Disease")]
    if q.get("food"):
        ents.append((q["food"], "Food"))
    return ents


def intent(q: dict) -> str:
    """一句话说清这道题问的是什么，防止改写跑题。"""
    t, sub, d = q["type"], q["subtype"], q["anchor"]
    if sub == "food_pair":
        return f"得了{d}能不能吃{q['food']}（是非题）"
    if t == "two_hop":
        what = {"check": "要做哪些检查", "symptom": "有哪些症状", "department": "该挂什么科"}[sub]
        return f"{d}的并发症{what}（问的是并发症，不是{d}本身）"
    if t == "symptoms":
        return f"同时有{'、'.join(q['symptoms'])}这几个症状，可能是什么病"
    if t == "backward":
        return f"哪些病和「{d}」有关（{q['question']}）"
    return q["question"]


def check(r: Rewrite, q: dict, link) -> str | None:
    """通过返回 None，否则返回原因（会发回给 DeepSeek）。"""
    said = {m.original: m.said for m in r.mentions}
    for name, label in key_entities(q):
        s = said.get(name)
        if not s:
            return f"mentions 里缺少「{name}」"
        if s not in r.question:
            return f"「{s}」没有原样出现在改写句里"
        matched = link(s, label).matched or link(s).matched
        if matched != name:
            return f"「{s}」在图谱里找不到「{name}」（链接到了 {matched}），请写得更接近原名"
    if len(r.question) > 80:
        return "太长了"
    if len(re.findall(r"[？?]", r.question)) > 1:          # 第二版有 7 题拆成了两问，答案范围跟着变了
        return "只能问一个问题，不要拆成两问"
    if any(w in r.question for w in BANNED):
        return f"不要用这些说法：{'、'.join(BANNED)}"
    return None


def rewrite_one(q: dict, llm_call, link) -> tuple[dict | None, list[str]]:
    rng = random.Random(f"rewrite|{q['id']}")
    entities = "、".join(f"「{n}」" for n, _ in key_entities(q))
    prompt = PROMPT.format(question=q["question"], intent=intent(q), style=rng.choice(STYLES), entities=entities)
    reasons = []
    for _ in range(ATTEMPTS):
        try:
            r = llm_call(prompt if not reasons else f"{prompt}\n\n上一次的改写不合格：{reasons[-1]}。请重新改写。")
        except Exception as e:                       # 结构化输出两次都没过
            reasons.append(f"输出格式错误：{type(e).__name__}")
            continue
        reason = check(r, q, link)
        if reason is None:
            return {**q, "id": f"{q['id']}-oral", "split": f"{q['split']}_oral", "template": q["question"],
                    "question": r.question.strip(), "mentions": {m.original: m.said for m in r.mentions}}, reasons
        reasons.append(reason)
    return None, reasons


def main():
    from bluemed.llm import structured_invoke
    from bluemed.tools.kg_tools import _link

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--frac", type=float, default=1.0, help="改写多少比例的题（按 id 固定抽样）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    use_memory_graph()
    from rl.eval import _serialize_embedding
    _serialize_embedding()                           # _link 会用到 m3e，多线程要加锁

    qs = [json.loads(l) for l in open(OUT_DIR / f"questions_{args.split}.jsonl")]
    qs = [q for q in qs if random.Random(f"oral|{q['id']}").random() < args.frac]
    out, rejected = OUT_DIR / f"questions_{args.split}_oral.jsonl", OUT_DIR / f"rewrite_{args.split}_rejected.jsonl"
    done = set()
    for p in (out, rejected):
        if p.exists():
            done |= {json.loads(l)["id"].removesuffix("-oral") for l in open(p)}
    todo = [q for q in qs if q["id"] not in done]
    print(f"{args.split}：抽中 {len(qs)} 题，已处理 {len(done)}，待改写 {len(todo)} → {out}")

    call = lambda prompt: structured_invoke(Rewrite, prompt)
    kept = dropped = 0
    with open(out, "a") as fo, open(rejected, "a") as fr, ThreadPoolExecutor(args.workers) as pool:
        for i, (q, (row, reasons)) in enumerate(zip(todo, pool.map(lambda q: rewrite_one(q, call, _link), todo))):
            if row:
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                fr.write(json.dumps({"id": q["id"], "question": q["question"], "reasons": reasons},
                                    ensure_ascii=False) + "\n")
                dropped += 1
            fo.flush(), fr.flush()
            print(f"  [{i + 1}/{len(todo)}] {'✓' if row else '✗'} {q['question'][:28]} → "
                  f"{row['question'] if row else reasons[-1]}", flush=True)
    print(f"保留 {kept}，丢掉 {dropped}")


if __name__ == "__main__":
    main()
