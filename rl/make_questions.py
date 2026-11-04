"""在图谱上自动出题：每道题都有程序能核对的标准答案（RL_PLAN.md 第 3 节）。

6 类题型：
  forward      一跳正向     痛风不能吃什么？                 → 忌吃食物集合
  backward     一跳反向     哪些病需要做尿酸检查？           → 疾病集合
  symptoms     多症状求交集 头痛又发烧可能是什么病？         → 同时有这些症状的疾病
  two_hop      两跳         糖尿病的并发症要做哪些检查？     → 各并发症的检查的并集
  attribute    属性         痛风是什么原因引起的？           → 证据里要有 (痛风, cause) 这个属性
  unanswerable 查不到       中耳炎饮食要注意什么？（图里没有）→ 应回答"证据不足"

每道题还带一组 oracle（标准做法：应该调哪些工具）。生成后在环境里把 oracle 执行一遍，
标准答案必须能从查到的证据里完整推出来，推不出来的题丢掉——保证每道题"用工具一定做得出来"。

防泄漏：按题目里的核心实体（anchor）划分 train / dev / test，同一个实体只出现在一个集合里。
只做模板出题；口语改写（rewrite.py，调 DeepSeek）是下一步。

用法：python -m rl.make_questions            # 输出到 data/rl/questions_{train,dev,test}.jsonl
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from rl.env import ExpertEnv
from rl.kg_memory import MemoryKG, get_memory_kg

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "rl"
TOOL_LIMIT = 15                     # kg_search 默认 limit：标准答案不能超过它，否则默认调用查不全

# ---------------------------------------------------------------- 模板
# 每组关系：(关系们, 带实体名的问法, 前面已有"我爸有{d}，"时的问法)
# 药物两种关系一起问：用户不会区分"推荐药"和"常用药"，标准答案取并集，模型要学会两种都查
FORWARD = {
    "symptom": (["HAS_SYMPTOM"], ["{d}有哪些症状？", "得了{d}会有什么表现？", "{d}一般有什么症状？"],
                ["一般会有哪些症状？", "会有什么表现？"]),
    "check": (["REQUIRES_CHECK"], ["{d}要做哪些检查？", "怀疑{d}应该查什么？", "{d}需要做什么检查？"],
              ["需要做哪些检查？", "医院一般会查什么？"]),
    "drug": (["RECOMMENDS_DRUG", "COMMONLY_USES_DRUG"], ["{d}吃什么药？", "{d}一般用什么药治疗？", "{d}患者常用哪些药？"],
             ["一般吃什么药？", "可以用哪些药？"]),
    "not_eat": (["SHOULD_NOT_EAT"], ["{d}不能吃什么？", "{d}饮食上要忌口什么？", "{d}患者要避免吃哪些食物？"],
                ["平时不能吃什么？", "饮食上要忌口什么？"]),
    "do_eat": (["SHOULD_EAT"], ["{d}适合吃什么？", "{d}宜吃哪些食物？", "{d}患者吃什么好？"],
               ["平时适合吃什么？", "吃什么比较好？"]),
    "recipe": (["RECOMMENDS_RECIPE"], ["{d}有哪些推荐的食谱？", "{d}可以做什么食疗？"],
               ["有什么推荐的食谱？"]),
    "department": (["TREATED_BY"], ["{d}挂什么科？", "{d}应该去哪个科室看？"],
                   ["应该挂什么科？", "去医院看哪个科？"]),
    "treatment": (["USES_TREATMENT"], ["{d}怎么治疗？", "{d}有哪些治疗方法？"],
                  ["一般怎么治疗？"]),
    "complication": (["ACCOMPANIES"], ["{d}会引起哪些并发症？", "{d}有什么并发症？"],
                     ["会有哪些并发症？"]),
}
CONTEXT = ["我爸有{d}，", "我妈得了{d}，", "孩子查出了{d}，", "我最近被诊断为{d}，", "朋友有{d}，"]
CONTEXT_RATE = 0.35                 # 这么多比例的题加上"我爸有……"之类的上下文（复用 v2 数据增强的思路）

BACKWARD = {
    "symptom": (["HAS_SYMPTOM"], "Symptom", ["哪些病会出现{t}？", "{t}可能是什么病引起的？"]),
    "check": (["REQUIRES_CHECK"], "Check", ["哪些病需要做{t}？", "{t}一般用来查什么病？"]),
    "drug": (["RECOMMENDS_DRUG", "COMMONLY_USES_DRUG"], "Drug", ["{t}能治哪些病？", "{t}一般用于什么病？"]),
    "not_eat": (["SHOULD_NOT_EAT"], "Food", ["哪些病的人不能吃{t}？", "什么病要忌口{t}？"]),
}
BACKWARD_MAX = 10                   # 反向题的答案最多几个疾病

SYMPTOM_TEMPLATES = ["最近{s}，可能是什么病？", "同时出现{s}，是什么病？", "{s}是什么病的症状？", "我{s}，这是怎么回事？"]
SYMPTOMS_MAX = 5                    # 求交集的答案最多几个疾病

TWO_HOP = {
    "check": ("REQUIRES_CHECK", ["{d}的并发症一般要做哪些检查？", "{d}可能引起的并发症需要查什么？"]),
    "symptom": ("HAS_SYMPTOM", ["{d}的并发症会有哪些症状？", "{d}引起的并发症有什么表现？"]),
    "department": ("TREATED_BY", ["{d}的并发症应该挂什么科？"]),
}
TWO_HOP_MAX_COMPLICATIONS = 3
TWO_HOP_MAX_ANSWER = 25

ATTRIBUTE = {
    "cause": ["{d}是什么原因引起的？", "为什么会得{d}？"],
    "prevent": ["{d}怎么预防？", "如何预防{d}？"],
    "easy_get": ["哪些人容易得{d}？", "{d}的易感人群有哪些？"],
    "get_way": ["{d}是怎么传播的？", "{d}会传染吗？"],
    "cure_lasttime": ["{d}要治多久？", "{d}的治疗周期多长？"],
    "cured_prob": ["{d}能治好吗？", "{d}的治愈率高吗？"],
    "cost_money": ["治{d}大概要花多少钱？"],
    "desc": ["{d}是什么病？", "介绍一下{d}。"],
}
PAIR_TEMPLATES = ["{d}能吃{y}吗？", "{d}患者可以吃{y}吗？", "得了{d}还能吃{y}吗？"]
PAIR_ANSWERABLE_RATE = 0.4          # 一跳正向题里这么多比例换成能回答的"能不能吃 Y"，数量和查不到的 food_pair 相当

TYPES = ["forward", "backward", "symptoms", "two_hop", "attribute", "unanswerable"]
SHARE = {"forward": 0.30, "backward": 0.10, "symptoms": 0.10, "two_hop": 0.25, "attribute": 0.10, "unanswerable": 0.15}
SIZES = {"train": 8000, "dev": 300, "test": 600}        # test 每类 100 题


def is_dirty(name: str) -> bool:
    """源数据里的脏名字：6 个"xx-测试"疾病和一个叫"测试"的症状（测试条目），355 个被截断成"xx..."的症状。
    图本身不改（要和线上 Neo4j 一致），只是不拿它们出题、也不让它们出现在标准答案里。"""
    return name.endswith("-测试") or name == "测试" or name.endswith("...")


def split_of(anchor: str) -> str:
    """按核心实体的哈希分组：80% train / 5% dev / 15% test。同一实体只会进一个集合，与随机种子无关。"""
    bucket = int(hashlib.md5(anchor.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "dev" if bucket < 85 else "test"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else "、".join(items[:-1]) + "和" + items[-1]


def _search(entity: str, relation: str, direction: str = "forward") -> dict:
    args = {"entity": entity, "relation": relation}
    if direction == "backward":
        args["direction"] = "backward"
    return {"name": "kg_search", "args": args}


# ---------------------------------------------------------------- 各题型的候选
class QuestionMaker:
    def __init__(self, kg: MemoryKG, rng: random.Random):
        self.kg, self.rng = kg, rng
        self.foods = sorted(kg.names("Food"))

    def _disease_question(self, d: str, plain: list[str], context: list[str]) -> str:
        if context and self.rng.random() < CONTEXT_RATE:
            return self.rng.choice(CONTEXT).format(d=d) + self.rng.choice(context)
        return self.rng.choice(plain).format(d=d)

    def forward(self, d: str) -> dict | None:
        options = []
        for key, (rels, plain, context) in FORWARD.items():
            lists = [self.kg.targets(d, r) for r in rels]
            if all(len(x) <= TOOL_LIMIT for x in lists) and any(lists):
                options.append((key, rels, plain, context, lists))
        if not options:
            return None
        key, rels, plain, context, lists = self.rng.choice(options)
        q = {"type": "forward", "anchor": d, "relations": rels, "subtype": key,
             "question": self._disease_question(d, plain, context),
             "gold": {"answer": sorted({t for x in lists for t in x})},
             "oracle": [_search(d, r) for r in rels]}
        return self._food_pair(d) or q

    def _food_pair(self, d: str) -> dict | None:
        """能回答的"能不能吃 Y"：Y 就在这个病的忌吃或宜吃里，答案是 [Y]（查到了这条关系）。
        和 unanswerable 里的 food_pair（Y 两边都没有）成对出现，否则模型学会"问能不能吃就说查不到"也能拿分。
        用按疾病名固定的随机数，不消耗主随机数，原有题目不受影响。"""
        local = random.Random(f"food_pair|{d}")
        foods = sorted(set(self.kg.targets(d, "SHOULD_NOT_EAT")) | set(self.kg.targets(d, "SHOULD_EAT")))
        if local.random() >= PAIR_ANSWERABLE_RATE or not foods:
            return None
        food = local.choice(foods)
        return {"type": "forward", "anchor": d, "relations": ["SHOULD_NOT_EAT", "SHOULD_EAT"], "subtype": "food_pair",
                "food": food, "question": local.choice(PAIR_TEMPLATES).format(d=d, y=food),
                "gold": {"answer": [food]},
                "oracle": [_search(d, "SHOULD_NOT_EAT"), _search(d, "SHOULD_EAT")]}

    def backward(self, t: str, key: str) -> dict | None:
        rels, _, templates = BACKWARD[key]
        lists = [self.kg.sources(t, r) for r in rels]
        answer = sorted({d for x in lists for d in x})
        if not 1 <= len(answer) <= BACKWARD_MAX:
            return None
        return {"type": "backward", "anchor": t, "relations": rels, "subtype": key,
                "question": self.rng.choice(templates).format(t=t),
                "gold": {"answer": answer}, "oracle": [_search(t, r, "backward") for r in rels]}

    def symptoms(self, d: str) -> dict | None:
        own = self.kg.targets(d, "HAS_SYMPTOM")
        if len(own) < 2:
            return None
        picked = self.rng.sample(own, min(len(own), self.rng.choice([2, 3])))
        answer = sorted(set.intersection(*(set(self.kg.sources(s, "HAS_SYMPTOM")) for s in picked)))
        if not 1 <= len(answer) <= SYMPTOMS_MAX:
            return None
        return {"type": "symptoms", "anchor": d, "relations": ["HAS_SYMPTOM"], "subtype": f"{len(picked)}_symptoms",
                "symptoms": picked, "question": self.rng.choice(SYMPTOM_TEMPLATES).format(s=_join(picked)),
                "gold": {"answer": answer},
                "oracle": [{"name": "find_diseases_by_symptoms", "args": {"symptoms": picked}}]}

    def two_hop(self, d: str) -> dict | None:
        complications = [c for c in self.kg.targets(d, "ACCOMPANIES") if c != d]       # 去掉自环
        if not 1 <= len(complications) <= TWO_HOP_MAX_COMPLICATIONS:
            return None
        options = []
        for key, (rel, templates) in TWO_HOP.items():
            lists = [self.kg.targets(c, rel) for c in complications]
            answer = sorted({t for x in lists for t in x})
            if all(len(x) <= TOOL_LIMIT for x in lists) and 1 <= len(answer) <= TWO_HOP_MAX_ANSWER:
                options.append((key, rel, templates, answer))
        if not options:
            return None
        key, rel, templates, answer = self.rng.choice(options)
        return {"type": "two_hop", "anchor": d, "relations": ["ACCOMPANIES", rel], "subtype": key,
                "complications": complications, "question": self.rng.choice(templates).format(d=d),
                "gold": {"answer": answer},
                "oracle": [_search(d, "ACCOMPANIES")] + [_search(c, rel) for c in complications]}

    def attribute(self, d: str) -> dict | None:
        props = self.kg.properties(d)
        fields = [f for f in ATTRIBUTE if props.get(f)]
        if not fields:
            return None
        f = self.rng.choice(fields)
        return {"type": "attribute", "anchor": d, "relations": [], "subtype": f,
                "question": self.rng.choice(ATTRIBUTE[f]).format(d=d),
                "gold": {"attribute": [d, f]},
                "oracle": [{"name": "get_disease_info", "args": {"disease": d, "fields": [f]}}]}

    def unanswerable(self, d: str) -> dict | None:
        """两种查不到：① 这个病在图里没有这类关系；② 问某个具体食物能不能吃，而图里的宜吃 / 忌吃都没提到它。"""
        if self.rng.random() < 0.5:
            missing = [(k, v) for k, v in FORWARD.items()
                       if k in ("drug", "not_eat", "do_eat", "complication") and not any(self.kg.targets(d, r) for r in v[0])]
            if not missing:
                return None
            key, (rels, plain, context) = self.rng.choice(missing)
            return {"type": "unanswerable", "anchor": d, "relations": rels, "subtype": f"no_{key}",
                    "question": self._disease_question(d, plain, context),
                    "gold": {"insufficient": True}, "oracle": [_search(d, r) for r in rels]}
        eat, not_eat = self.kg.targets(d, "SHOULD_EAT"), self.kg.targets(d, "SHOULD_NOT_EAT")
        if not (eat or not_eat):
            return None
        food = self.rng.choice([f for f in self.foods if f not in set(eat) | set(not_eat)])
        return {"type": "unanswerable", "anchor": d, "relations": ["SHOULD_NOT_EAT", "SHOULD_EAT"], "subtype": "food_pair",
                "food": food, "question": self.rng.choice(PAIR_TEMPLATES).format(d=d, y=food),
                "gold": {"insufficient": True},
                "oracle": [_search(d, "SHOULD_NOT_EAT"), _search(d, "SHOULD_EAT")]}


# ---------------------------------------------------------------- 校验：执行 oracle，标准答案必须能从证据推出
def oracle_answer(q: dict, env: ExpertEnv) -> dict:
    """按题型从 oracle 查到的证据里"推出"答案，格式和 gold 相同。"""
    ev = list(env.evidence.values())
    t = q["type"]
    if t == "forward" and q["subtype"] == "food_pair":
        hit = any(e.subject == q["anchor"] and e.object == q["food"] and e.predicate in q["relations"] for e in ev)
        return {"answer": [q["food"]]} if hit else {}
    if t == "forward":
        return {"answer": sorted({e.object for e in ev if e.subject == q["anchor"] and e.predicate in q["relations"]})}
    if t == "backward":
        return {"answer": sorted({e.subject for e in ev if e.object == q["anchor"] and e.predicate in q["relations"]})}
    if t == "symptoms":
        return {"answer": sorted({e.subject for e in ev if e.score == 1.0})}     # 命中全部症状的疾病
    if t == "two_hop":
        rel = q["relations"][1]
        return {"answer": sorted({e.object for e in ev if e.subject in q["complications"] and e.predicate == rel})}
    if t == "attribute":
        d, f = q["gold"]["attribute"]
        return {"attribute": [d, f]} if any(e.kind == "attribute" and e.subject == d and e.predicate == f for e in ev) else {}
    # unanswerable：oracle 查完后，证据里不能有能回答问题的东西
    if q["subtype"] == "food_pair":
        return {"insufficient": True} if all(e.object != q["food"] for e in ev) else {}
    return {"insufficient": True} if not ev else {}


def _has_dirty(q: dict) -> bool:
    mentioned = [q["anchor"], q.get("food", "")] + q.get("symptoms", []) + q.get("complications", [])
    return any(is_dirty(n) for n in mentioned + q["gold"].get("answer", []))


def verify(q: dict) -> bool:
    env = ExpertEnv()
    results = env.step(q["oracle"])
    return all(r.status == "ok" for r in results) and oracle_answer(q, env) == q["gold"]


# ---------------------------------------------------------------- 主流程
def build(seed: int = 0) -> dict[str, list[dict]]:
    kg = get_memory_kg()
    rng = random.Random(seed)
    maker = QuestionMaker(kg, rng)

    # 每个集合、每种题型的候选 anchor（按 split_of 分好组）
    anchors = defaultdict(lambda: defaultdict(list))
    for d in kg.diseases:
        for t in ["forward", "symptoms", "two_hop", "attribute", "unanswerable"]:
            anchors[split_of(d)][t].append((d, None))
    for key, (_, label, _) in BACKWARD.items():
        for t in kg.names(label):
            anchors[split_of(t)]["backward"].append((t, key))

    data = {}
    for split, total in SIZES.items():
        questions, seen = [], set()
        for qtype in TYPES:
            want = round(total * SHARE[qtype]) if split != "test" else total // len(TYPES)
            pool = anchors[split][qtype][:]
            rng.shuffle(pool)
            got = 0
            for anchor, key in pool:
                if got >= want:
                    break
                q = maker.backward(anchor, key) if qtype == "backward" else getattr(maker, qtype)(anchor)
                if q is None or (qtype, anchor) in seen or _has_dirty(q) or not verify(q):
                    continue
                seen.add((qtype, anchor))
                q["min_calls"] = len(q["oracle"])
                questions.append(q)
                got += 1
            if got < want:
                print(f"  ⚠️ {split}/{qtype}：只出了 {got} 题（想要 {want}）")
        rng.shuffle(questions)
        for i, q in enumerate(questions):
            q["id"] = f"{split}-{i:05d}"
            q["split"] = split
        data[split] = questions
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    data = build(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for split, questions in data.items():
        path = args.out / f"questions_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps({"id": q.pop("id"), "split": q.pop("split"), **q}, ensure_ascii=False) + "\n")
        types = Counter(q["type"] for q in questions)
        print(f"{split}: {len(questions)} 题 {dict(types)} → {path}")
    anchors = {s: {q["anchor"] for q in qs} for s, qs in data.items()}
    assert not (anchors["train"] & anchors["test"]) and not (anchors["train"] & anchors["dev"]), "anchor 泄漏"


if __name__ == "__main__":
    main()
