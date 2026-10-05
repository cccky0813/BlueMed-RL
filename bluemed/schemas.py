"""全项目共用的数据结构：图谱的类型体系 + 证据格式。工具、Agent、Reviewer 都从这里引用，保证只有一份定义。"""

import hashlib
from typing import Literal, get_args

from pydantic import BaseModel, Field

# 图谱里实际存在的节点类型和关系类型（与 v1 建图时的 schema 一致）
NodeType = Literal["Disease", "Symptom", "Drug", "Food", "Recipe", "Check", "Department", "Treatment", "Category"]
Relation = Literal[
    "HAS_SYMPTOM", "REQUIRES_CHECK", "RECOMMENDS_DRUG", "COMMONLY_USES_DRUG", "SHOULD_EAT",
    "SHOULD_NOT_EAT", "RECOMMENDS_RECIPE", "TREATED_BY", "USES_TREATMENT", "BELONGS_TO", "ACCOMPANIES",
]
DiseaseField = Literal["desc", "cause", "prevent", "easy_get", "get_way", "cure_lasttime", "cured_prob", "cost_money"]

NODE_TYPES = set(get_args(NodeType))
RELATIONS = set(get_args(Relation))

RELATION_CN = {
    "HAS_SYMPTOM": "症状", "REQUIRES_CHECK": "需要的检查", "RECOMMENDS_DRUG": "推荐药物",
    "COMMONLY_USES_DRUG": "常用药物", "SHOULD_EAT": "宜吃食物", "SHOULD_NOT_EAT": "忌吃食物",
    "RECOMMENDS_RECIPE": "推荐食谱", "TREATED_BY": "就诊科室", "USES_TREATMENT": "治疗方法",
    "BELONGS_TO": "所属分类", "ACCOMPANIES": "并发症",
}
FIELD_CN = {
    "desc": "简介", "cause": "病因", "prevent": "预防", "easy_get": "易感人群", "get_way": "传播途径",
    "cure_lasttime": "治疗周期", "cured_prob": "治愈率", "cost_money": "治疗费用",
}


class Evidence(BaseModel):
    """一条证据：一个三元组，或一个疾病属性。id 由内容决定，同一条事实无论被哪个工具查到，id 都相同。"""
    id: str
    kind: Literal["triple", "attribute"]
    subject: str
    predicate: str          # 关系类型（triple）或属性名（attribute）
    object: str
    hop: int = 1
    score: float = 0.0

    @classmethod
    def make(cls, kind, subject, predicate, obj, **kw) -> "Evidence":
        digest = hashlib.sha1(f"{kind}|{subject}|{predicate}|{obj}".encode()).hexdigest()[:6]
        return cls(id=f"E{digest}", kind=kind, subject=subject, predicate=predicate, object=obj, **kw)

    def to_text(self) -> str:
        """给模型看的一行文字。"""
        if self.kind == "attribute":
            return f"[{self.id}] {self.subject}的{FIELD_CN.get(self.predicate, self.predicate)}：{self.object}"
        return f"[{self.id}] {self.subject} -{self.predicate}-> {self.object}"


class LinkResult(BaseModel):
    query: str
    matched: str | None = Field(description="图谱中的规范名，找不到为 None")
    node_type: str | None = None
    score: float = 0.0
    method: Literal["exact", "fuzzy", "none"] = "none"
