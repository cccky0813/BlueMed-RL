"""图谱的内存版：RL 训练要调用工具几十万次，不能每次都查 Neo4j。

从 data/medical.json 按 v1 的建图规则（BlueMedicial/build_graph.py）在内存里建同一张图，
实现和 bluemed.kg.Neo4jBackend 相同的 6 种查询。工具逻辑不重写：

    from rl.kg_memory import use_memory_graph
    use_memory_graph()          # 之后 bluemed.tools.kg_tools 里的 5 个工具都查内存

和 Neo4j 版的差别只在"不确定"的地方：Neo4j 的 LIMIT 没有 ORDER BY，超出上限时返回哪几条由数据库决定；
这里固定按 medical.json 里的出现顺序，结果可复现。tests/test_kg_memory.py 验证两者在其余情况下一致。
"""

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import get_args

from bluemed.kg import MAX_ROWS
from bluemed.schemas import NodeType

DATA = Path(__file__).resolve().parents[1] / "data" / "medical.json"

# medical.json 字段 -> (目标节点类型, 关系类型)，与 build_graph.py 一致
FIELDS = {
    "category": ("Category", "BELONGS_TO"),
    "symptom": ("Symptom", "HAS_SYMPTOM"),
    "cure_department": ("Department", "TREATED_BY"),
    "cure_way": ("Treatment", "USES_TREATMENT"),
    "check": ("Check", "REQUIRES_CHECK"),
    "recommand_drug": ("Drug", "RECOMMENDS_DRUG"),
    "common_drug": ("Drug", "COMMONLY_USES_DRUG"),
    "do_eat": ("Food", "SHOULD_EAT"),
    "not_eat": ("Food", "SHOULD_NOT_EAT"),
    "recommand_eat": ("Recipe", "RECOMMENDS_RECIPE"),
    "acompany": ("Disease", "ACCOMPANIES"),
}
TARGET_LABEL = {rel: label for label, rel in FIELDS.values()}
DISEASE_PROPS = ["desc", "prevent", "cause", "get_prob", "easy_get", "get_way",
                 "cure_lasttime", "cured_prob", "cost_money"]
# 同一个名字可能同时是疾病和症状等多种节点；Neo4j 里 LIMIT 1 取哪个不确定，这里固定按这个顺序
LABEL_ORDER = list(get_args(NodeType))
CONTAINS_LIMIT = 50


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [v.strip() for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


class MemoryKG:
    """内存图谱 + Neo4jBackend 的 6 种查询。"""

    def __init__(self, path: Path = DATA):
        self.labels: dict[str, list[str]] = defaultdict(list)                       # 名字 -> 节点类型（按 LABEL_ORDER）
        self.props: dict[str, dict] = {}                                             # 疾病 -> 文字属性
        self.out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))   # 名字 -> 关系 -> 目标
        self.inc: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))   # 名字 -> 关系 -> 来源
        self.diseases: list[str] = []                                                # medical.json 里有记录的疾病
        self._load(path)

    # ------------------------------------------------------------------ 建图
    def _add_node(self, name: str, label: str) -> None:
        if label not in self.labels[name]:
            self.labels[name].append(label)
            self.labels[name].sort(key=LABEL_ORDER.index)

    def _load(self, path: Path) -> None:
        seen_edges = set()
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            d = r["name"].strip()
            self._add_node(d, "Disease")
            if d not in self.props:
                self.diseases.append(d)
            # 同名疾病出现多次时，后面的属性覆盖前面的（对应 Cypher 的 SET d += props）
            props = {k: r[k] if isinstance(r.get(k), str) else "、".join(_as_list(r.get(k)))
                     for k in DISEASE_PROPS if r.get(k)}
            self.props[d] = {**self.props.get(d, {}), **props}
            for field, (label, rel) in FIELDS.items():
                for t in _as_list(r.get(field)):
                    if label == "Category" and t == "疾病百科":
                        continue
                    self._add_node(t, label)
                    if (d, rel, t) not in seen_edges:                    # MERGE：同一条边只建一次
                        seen_edges.add((d, rel, t))
                        self.out[d][rel].append(t)
                        self.inc[t][rel].append(d)

    # ------------------------------------------------------------------ 6 种查询（与 Neo4jBackend 同名同返回格式）
    def exact(self, name: str, labels: list[str]) -> dict | None:
        for label in self.labels.get(name, []):
            if label in labels:
                return {"name": name, "label": label}
        return None

    def contains(self, name: str, labels: list[str]) -> list[dict]:
        rows = [{"name": n, "label": label} for n, ls in self.labels.items()
                if len(n) >= 2 and (name in n or n in name) for label in ls if label in labels]
        # Neo4j 版是任意 50 个；这里取长度最接近的 50 个，结果固定，也更可能包含正确实体
        rows.sort(key=lambda r: (abs(len(r["name"]) - len(name)), r["name"]))
        return rows[:CONTAINS_LIMIT]

    def relation(self, name: str, relation: str, direction: str, limit: int) -> list[dict]:
        if direction == "forward":
            return [{"s": name, "t": t} for t in self.out.get(name, {}).get(relation, [])[:limit]]
        return [{"s": s, "t": name} for s in self.inc.get(name, {}).get(relation, [])[:limit]]

    def properties(self, disease: str) -> dict:
        if "Disease" not in self.labels.get(disease, []):
            return {}
        return {"name": disease, **self.props.get(disease, {})}          # Neo4j 的 properties() 也包含 name

    def neighbors(self, name: str, per_rel: int) -> list[dict]:
        rows = []
        for rel, targets in self.out.get(name, {}).items():
            rows += [{"rel": rel, "s": name, "t": t, "m": t, "label": TARGET_LABEL[rel]} for t in targets[:per_rel]]
        for rel, sources in self.inc.get(name, {}).items():
            # 自环（某病把自己列为并发症）在 Neo4j 里只算出边一次；同名的疾病→症状不是自环，两头都算
            sources = [s for s in sources if not (s == name and TARGET_LABEL[rel] == "Disease")]
            rows += [{"rel": rel, "s": s, "t": name, "m": s, "label": "Disease"} for s in sources[:per_rel]]
        return rows[:MAX_ROWS]

    def diseases_by_symptoms(self, symptoms: list[str], limit: int) -> list[dict]:
        hits: dict[str, list[str]] = defaultdict(list)
        for s in dict.fromkeys(symptoms):
            for d in self.inc.get(s, {}).get("HAS_SYMPTOM", []):
                hits[d].append(s)
        rows = sorted(hits.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:limit]
        return [{"d": d, "hit": hit} for d, hit in rows]

    # ------------------------------------------------------------------ 出题用的辅助查询
    def targets(self, name: str, relation: str) -> list[str]:
        return list(self.out.get(name, {}).get(relation, []))

    def sources(self, name: str, relation: str) -> list[str]:
        return list(self.inc.get(name, {}).get(relation, []))

    def names(self, label: str) -> list[str]:
        return [n for n, ls in self.labels.items() if label in ls]

    def stats(self) -> tuple[dict[str, int], dict[str, int]]:
        """（每种节点的个数，每种关系的条数），用来和 Neo4j 对账。"""
        nodes = defaultdict(int)
        for ls in self.labels.values():
            for label in ls:
                nodes[label] += 1
        edges = defaultdict(int)
        for rels in self.out.values():
            for rel, ts in rels.items():
                edges[rel] += len(ts)
        return dict(nodes), dict(edges)


@lru_cache(maxsize=1)
def get_memory_kg() -> MemoryKG:
    """整个进程共用一份（建图约几秒）。"""
    return MemoryKG()


def use_memory_graph() -> MemoryKG:
    """让 bluemed 的图谱工具改查内存图谱，返回这张图。"""
    from bluemed.tools import kg_tools
    kg = get_memory_kg()
    kg_tools.set_backend(kg)
    return kg
