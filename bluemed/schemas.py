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


class Answer(BaseModel):
    """Answer Generator 的结构化输出。"""
    answer: str = Field(description="给用户的回答。每个事实性陈述后面标注引用的证据编号，如 [E3c9314]；可分点")
    citations: list[str] = Field(description="回答中引用的所有证据编号，如 [\"E3c9314\"]")
    safety_notice: str = Field(description="安全提示：涉及诊断时建议就医，涉及用药时提醒遵医嘱；无需提示时为空字符串")
    insufficient: bool = Field(description="证据不足以回答用户问题时为 true")


class Review(BaseModel):
    """Answer Reviewer 的审核结论。"""
    verdict: Literal["pass", "rewrite", "need_evidence"] = Field(
        description="pass：可以发布；rewrite：回答有证据不支持的内容或表述问题，需要重写；need_evidence：证据不够回答问题，需要补充检索")
    reason: str = Field(description="判断理由，具体指出问题所在")
    missing: str = Field("", description="verdict 为 need_evidence 时，说明还需要查什么")
