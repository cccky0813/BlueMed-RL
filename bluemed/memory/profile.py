"""长期记忆：用户健康档案，存在 LangGraph Store 里，按 user_id 区分，跨会话（thread）共享。

写入策略：只记 Parser 从用户原话中抽取到的明确信息（咨询对象、已患疾病、过敏），不记模型推断的内容。
合并策略：疾病 / 过敏取并集（去重，保持顺序）；咨询对象以最新一次为准。
"""

from langgraph.store.base import BaseStore

from bluemed.schemas import ProfileUpdate

NAMESPACE = "profile"
EMPTY = {"subject": "", "conditions": [], "allergies": []}


def load_profile(store: BaseStore | None, user_id: str) -> dict:
    if store is None:
        return dict(EMPTY)
    item = store.get((NAMESPACE, user_id), "health")
    return {**EMPTY, **(item.value if item else {})}


def merge_profile(old: dict, update: ProfileUpdate) -> dict:
    return {
        "subject": update.subject or old.get("subject", ""),
        "conditions": list(dict.fromkeys(old.get("conditions", []) + update.conditions)),
        "allergies": list(dict.fromkeys(old.get("allergies", []) + update.allergies)),
    }


def save_profile(store: BaseStore | None, user_id: str, profile: dict) -> None:
    if store is not None:
        store.put((NAMESPACE, user_id), "health", profile)


def describe(profile: dict) -> str:
    parts = []
    if profile.get("subject"):
        parts.append(f"咨询对象：{profile['subject']}")
    if profile.get("conditions"):
        parts.append(f"已知疾病：{'、'.join(profile['conditions'])}")
    if profile.get("allergies"):
        parts.append(f"过敏：{'、'.join(profile['allergies'])}")
    return "；".join(parts) or "（暂无）"
