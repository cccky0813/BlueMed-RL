"""难测评集：A 组在模板题上已经 0.94，要换一批同分布以外的题才拉得开 A / B / C（RL_PLAN.md 第 7.1 节）。

都从 test 集出题（疾病不在 train 里；lay 不够时加 dev），标准答案、oracle 沿用原题，只换问法。三类，每类约 100 题：
  free        没见过的问法：只给 DeepSeek 题意，不给模板原文，让它自由提问
  lay         口语化实体：实体换成俗称（肚子疼 → 腹痛）。要求 _link 链接不回原实体（模型得自己换成规范名），
              但另一次独立调用的 DeepSeek 能从俗称猜回原名（保证题目可解）
  distractor  干扰实体：问题里多一个无关的病作背景（"我爸有高血压，最近查出痛风……"），考会不会查错对象

  python -m rl.hard_set            # → data/rl/questions_hard_{free,lay,distractor}.jsonl
  python -m rl.eval --model deepseek --split hard_lay
"""

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from rl.kg_memory import use_memory_graph
from rl.make_questions import OUT_DIR
from rl.rewrite import BANNED, Rewrite, intent, key_entities

PER_MODE = 100
ATTEMPTS = 2
MODES = ["free", "lay", "distractor"]
DISEASE_TYPES = ("forward", "two_hop", "attribute", "unanswerable")     # 主实体是疾病的题型（干扰实体只加病）
LAY_MAX_NAME = 6                                                        # 干扰病名最长几个字

FREE_PROMPT = """你是一个普通用户，要向医疗助手提一个问题。不要照抄任何模板，用你自己的说法问。

要问的事：{intent}
必须提到的实体（原样写出，不能改字）：{entities}
提问方式：{style}

要求：只问这一件事，不多问、不少问；是非题还是是非题；不加新的症状、疾病、食物；一句或两句话，不超过 60 字；只能有一个问号；
不要用"急死人了""病友们"之类的套话。mentions 里写出每个实体在问题里的原文片段。"""

FREE_STYLES = ["先说自己的情况再问", "很短，像发微信", "反问句或带转折（\"听说……是真的吗？那……\"不行，只能问一件事）",
               "用\"请问\"开头、比较正式", "倒装或口头禅（\"……是吧\"\"……来着\"）", "把问题放在句首，后面补一句背景"]

LAY_PROMPT = """把下面这个医学术语换成普通人日常会说的俗称或大白话（例如 腹痛 → 肚子疼，糖尿病 → 血糖高的病，呼吸困难 → 喘不上气）。

术语：{name}（{label}）

要求：俗称必须是大多数人一听就知道指它的说法；不能只是去掉或加上一两个字；如果这个术语没有常见的俗称，has_lay 填 false。"""

GUESS_PROMPT = """一个病人说"{lay}"，最可能指的是哪个{label}？只写一个最规范的医学名称。"""

DISTRACTOR_PROMPT = """你是一个普通用户。请用自然的口语提一个问题，问题里要先顺带提到另一个病当背景（比如家人有这个病，或者自己以前得过），但真正要问的只有一件事。

背景里顺带提到的病（原样写出）：「{distractor}」
真正要问的事：{intent}
真正问的实体（原样写出）：{entities}

要求：真正的问题只能问上面那件事，背景里的病只是顺带一提，不能问它；只能有一个问号；不超过 70 字。mentions 里写出每个实体（包括背景里的病）在问题里的原文片段。"""

LABEL_CN = {"Disease": "疾病", "Symptom": "症状", "Check": "检查项目", "Drug": "药品", "Food": "食物"}


class Lay(BaseModel):
    has_lay: bool
    lay: str = Field("", description="俗称")


class Guess(BaseModel):
    name: str


def _basic_check(question: str, max_len: int = 80) -> str | None:
    if len(question) > max_len:
        return "太长了"
    if len(re.findall(r"[？?]", question)) > 1:
        return "只能问一个问题"
    if any(w in question for w in BANNED):
        return f"不要用这些说法：{'、'.join(BANNED)}"
    return None


def _mentions_ok(r: Rewrite, names: list[tuple[str, str]], link) -> str | None:
    said = {m.original: m.said for m in r.mentions}
    for name, label in names:
        s = said.get(name)
        if not s or s not in r.question:
            return f"「{name}」没有原样出现在问题里"
        if (link(s, label).matched or link(s).matched) != name:
            return f"「{s}」在图谱里对不上「{name}」，请原样写出"
    return None


def make_free(q, call, link, rng):
    names = key_entities(q)
    prompt = FREE_PROMPT.format(intent=intent(q), entities="、".join(f"「{n}」" for n, _ in names),
                                style=rng.choice(FREE_STYLES))
    reasons = []
    for _ in range(ATTEMPTS):
        r = call(Rewrite, prompt + (f"\n\n上一次不合格：{reasons[-1]}" if reasons else ""))
        reason = _basic_check(r.question) or _mentions_ok(r, names, link)
        if not reason:
            return {"question": r.question.strip(), "mentions": {m.original: m.said for m in r.mentions}}, reasons
        reasons.append(reason)
    return None, reasons


def make_lay(q, call, link, rng):
    """把一个关键实体换成俗称（多症状题换其中一个症状）；俗称要 _link 不回去、但 DeepSeek 能猜回来。"""
    names = key_entities(q)
    target, label = rng.choice(names)
    lay = call(Lay, LAY_PROMPT.format(name=target, label=LABEL_CN.get(label, label)))
    if not lay.has_lay or not lay.lay or lay.lay == target:
        return None, ["没有常见俗称"]
    if (link(lay.lay, label).matched or link(lay.lay).matched) == target:
        return None, [f"「{lay.lay}」直接就能链接回去，不算难"]
    if link(lay.lay).method == "exact":          # 俗称本身是图谱里另一个实体（红眼病、肠胃炎……），查出来是它的数据，题意有歧义
        return None, [f"「{lay.lay}」本身是图谱里另一个实体，有歧义"]
    guess = call(Guess, GUESS_PROMPT.format(lay=lay.lay, label=LABEL_CN.get(label, label))).name.strip()
    if (link(guess, label).matched or link(guess).matched) != target:
        return None, [f"「{lay.lay}」猜回来是「{guess}」，不是「{target}」，题目不可解"]
    rest = [(n, l) for n, l in names if n != target]
    entities = "、".join([f"「{lay.lay}」（指{target}，必须用这个俗称，不能写{target}）"] + [f"「{n}」" for n, _ in rest])
    prompt = FREE_PROMPT.format(intent=intent(q), entities=entities, style=rng.choice(FREE_STYLES))
    reasons = []
    for _ in range(ATTEMPTS):
        r = call(Rewrite, prompt + (f"\n\n上一次不合格：{reasons[-1]}" if reasons else ""))
        reason = (_basic_check(r.question) or _mentions_ok(r, rest, link)
                  or (None if lay.lay in r.question and target not in r.question else f"必须写「{lay.lay}」，不能写「{target}」"))
        if not reason:
            return {"question": r.question.strip(), "lay": {target: lay.lay}, "guess": guess}, reasons
        reasons.append(reason)
    return None, reasons


def make_distractor(q, call, link, rng, diseases):
    names = key_entities(q)
    distractor = rng.choice([d for d in diseases if d != q["anchor"] and d not in q["anchor"] and q["anchor"] not in d])
    prompt = DISTRACTOR_PROMPT.format(distractor=distractor, intent=intent(q),
                                      entities="、".join(f"「{n}」" for n, _ in names))
    reasons = []
    for _ in range(ATTEMPTS):
        r = call(Rewrite, prompt + (f"\n\n上一次不合格：{reasons[-1]}" if reasons else ""))
        reason = _basic_check(r.question, 90) or _mentions_ok(r, names + [(distractor, "Disease")], link)
        if not reason:
            return {"question": r.question.strip(), "distractor": distractor}, reasons
        reasons.append(reason)
    return None, reasons


def pool(mode: str, qs: list[dict]) -> list[dict]:
    """每类题的候选：lay 不限题型（俗称多在常见病、症状上，按需多抽）；distractor 只用主实体是疾病的题。"""
    if mode == "distractor":
        qs = [q for q in qs if q["type"] in DISEASE_TYPES]
    return sorted(qs, key=lambda q: random.Random(f"hard|{mode}|{q['id']}").random())


def main():
    from bluemed.llm import structured_invoke
    from bluemed.tools.kg_tools import _link
    from rl.eval import _serialize_embedding

    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--per-mode", type=int, default=PER_MODE)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    use_memory_graph()
    _serialize_embedding()
    test = [json.loads(l) for l in open(OUT_DIR / "questions_test.jsonl")]
    dev = [json.loads(l) for l in open(OUT_DIR / "questions_dev.jsonl")]
    # 干扰病只取名字短的（≤ 6 字），像用户会顺口提到的病；试采时抽到"Austin型幼儿脑硫脂病"这类，太不自然
    diseases = sorted({q["anchor"] for q in test if q["type"] in DISEASE_TYPES and len(q["anchor"]) <= LAY_MAX_NAME})
    call = lambda schema, prompt: structured_invoke(schema, prompt)

    for mode in args.modes:
        out = OUT_DIR / f"questions_hard_{mode}.jsonl"
        done = [json.loads(l) for l in open(out)] if out.exists() else []
        have = {r["id"].rsplit("-", 1)[0] for r in done}
        # lay 通过率只有约 10%（多数术语没有常见俗称），test 600 题不够，再加上 dev（疾病同样不在 train 里）
        cand = [q for q in pool(mode, test) + (pool(mode, dev) if mode == "lay" else []) if q["id"] not in have]
        kept, tried, why = len(done), 0, {}

        def one(q):
            rng = random.Random(f"hard|{mode}|{q['id']}")
            try:
                if mode == "free":
                    return make_free(q, call, _link, rng)
                if mode == "lay":
                    return make_lay(q, call, _link, rng)
                return make_distractor(q, call, _link, rng, diseases)
            except Exception as e:
                return None, [f"接口或格式错误：{type(e).__name__}"]

        with open(out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
            while kept < args.per_mode and cand:                 # lay 的通过率低，一批一批地抽，够数就停
                batch, cand = cand[:args.workers * 4], cand[args.workers * 4:]
                for q, (extra, reasons) in zip(batch, ex.map(one, batch)):
                    tried += 1
                    if extra and kept < args.per_mode:
                        row = {**q, **extra, "id": f"{q['id']}-{mode}", "split": f"hard_{mode}", "template": q["question"]}
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        kept += 1
                        print(f"  ✓ [{mode} {kept}] {q['type']:12s} {q['question'][:24]} → {row['question']}", flush=True)
                    elif reasons:
                        key = re.sub(r"「[^」]*」", "X", reasons[-1])[:16]       # 去掉具体实体，按原因归类
                        why[key] = why.get(key, 0) + 1
                f.flush()
        print(f"{mode}：留下 {kept}，这次试了 {tried} 题；没通过的原因：{why}")


if __name__ == "__main__":
    main()
