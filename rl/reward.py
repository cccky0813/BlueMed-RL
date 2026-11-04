"""奖励函数：给一条做完的 Expert 轨迹打分（RL_PLAN.md 第 4 节）。

采老师轨迹时用它筛数据（拒绝采样），评测时用它算分，GRPO 时用它算优势——三处用同一个函数。

输入：题目（make_questions.py 生成，带 gold / min_calls）、模型最后一条回复、做题用的 ExpertEnv
     （env 里记着查到的证据、调用次数、非法调用次数）。
输出：Reward，total 是总分，其余字段是拆解，训练时画曲线、查"钻空子"用（第 6.3 节）。

计分规则：
  格式不对（最后一行不是合法 JSON / 没给出最终回复）       → −1，直接结束
  内容分（按题型）：
    能回答的题   F1(答案, 标准答案)；答案里没在证据中出现过的实体（"无据"）一律算错
    属性题       证据里查到了 (疾病, 字段) 这个属性 → 1，否则 0
    查不到的题   查过之后回答 insufficient → 1；一次都没查就说查不到 → 0；给了答案 → −0.5（编造比不答更糟）
    能答却说查不到 → 0
  扣分：每次非法调用 −0.1；调用（含重复调用）超过"最少调用 + 2"的部分每次 −0.05；扣分合计最多 0.5
总分范围 [−1, 1]。

为什么格式错是 −1 而不是计划初稿里的 0：内容分最低是 −0.5，再加扣分可以到 −1。如果格式错只记 0，
没把握的题故意不按格式输出反而比认真作答得分高，模型会学会"交白卷"。所以格式错必须是最低分。
"""

import json
from dataclasses import asdict, dataclass

from rl.env import ExpertEnv

INVALID_PENALTY = 0.1
EXTRA_CALL_PENALTY = 0.05
FREE_EXTRA_CALLS = 2                # 比最少调用多调 2 次以内不扣分（例如先 link_entity 确认一下实体）
MAX_PENALTY = 0.5
FABRICATE = -0.5                    # 查不到的题给了答案
FORMAT_FAIL = -1.0


@dataclass
class Reward:
    total: float
    format_ok: bool
    content: float = 0.0            # 扣分前的内容分
    penalty: float = 0.0
    status: str = ""                # sufficient / insufficient（模型的判断）
    n_answer: int = 0
    n_gold: int = 0
    n_ungrounded: int = 0           # 答案里没在证据中出现过的实体数（"不查就答"的信号）
    precision: float = 0.0
    recall: float = 0.0
    calls: int = 0
    invalid: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def parse_final(text: str | None) -> dict | None:
    """取最后一个非空行解析成 JSON；不合格式返回 None。允许外面包一层 ```json 代码块。"""
    if not text:
        return None
    lines = [l.strip() for l in text.strip().splitlines() if l.strip() and not l.strip().startswith("```")]
    if not lines:
        return None
    try:
        obj = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    status = obj.get("status")
    if status == "insufficient":
        return {"status": status, "answer": []}
    answer = obj.get("answer")
    if status != "sufficient" or not isinstance(answer, list) or not all(isinstance(a, str) for a in answer):
        return None
    return {"status": status, "answer": sorted({a.strip() for a in answer if a.strip()})}


def evidence_entities(env: ExpertEnv) -> set[str]:
    """证据里出现过的所有实体名（三元组的主语和宾语）。答案只能从这里面取。"""
    return {n for e in env.evidence.values() if e.kind == "triple" for n in (e.subject, e.object)}


def _f1(answer: list[str], gold: list[str], grounded: set[str]) -> tuple[float, float, float]:
    tp = len(set(answer) & set(gold) & grounded)          # 无据的实体即使恰好在标准答案里也不算对
    if tp == 0:
        return 0.0, 0.0, 0.0
    p, r = tp / len(answer), tp / len(gold)
    return 2 * p * r / (p + r), p, r


def _penalty(env: ExpertEnv, min_calls: int) -> float:
    duplicates = sum(r.status == "duplicate" for r in env.history)
    extra = max(0, env.calls - env.invalid + duplicates - (min_calls + FREE_EXTRA_CALLS))
    return min(MAX_PENALTY, INVALID_PENALTY * env.invalid + EXTRA_CALL_PENALTY * extra)


def score(q: dict, final_text: str | None, env: ExpertEnv) -> Reward:
    """q：题目；final_text：模型最后一条（不再调用工具的）回复，没有就传 None；env：做这道题用的环境。"""
    final = parse_final(final_text)
    if final is None:
        return Reward(total=FORMAT_FAIL, format_ok=False, calls=env.calls, invalid=env.invalid)

    gold = q["gold"]
    r = Reward(total=0.0, format_ok=True, status=final["status"], n_answer=len(final["answer"]),
               calls=env.calls, invalid=env.invalid)
    if "insufficient" in gold:
        if final["status"] == "insufficient":
            r.content = 1.0 if env.calls > env.invalid else 0.0       # 不查就说查不到，是蒙的
        else:
            r.content = FABRICATE if final["answer"] else 0.0
    elif final["status"] == "insufficient":
        r.content = 0.0
    elif "attribute" in gold:
        d, f = gold["attribute"]
        r.content = float(any(e.kind == "attribute" and e.subject == d and e.predicate == f
                              for e in env.evidence.values()))
    else:
        grounded = evidence_entities(env)
        r.n_gold = len(gold["answer"])
        r.n_ungrounded = sum(a not in grounded for a in final["answer"])
        r.content, r.precision, r.recall = _f1(final["answer"], gold["answer"], grounded)

    r.penalty = _penalty(env, q["min_calls"])
    r.total = round(r.content - r.penalty, 4)
    return r


def final_json(answer: list[str] | None) -> str:
    """把答案拼成标准的最后一行（None 表示查不到）。测试和 oracle 校准用。"""
    if answer is None:
        return json.dumps({"status": "insufficient", "missing": "图谱中没有相关信息"}, ensure_ascii=False)
    return json.dumps({"status": "sufficient", "answer": answer}, ensure_ascii=False)

