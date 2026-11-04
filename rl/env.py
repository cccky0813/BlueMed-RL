"""Expert 的工具执行环境：采老师轨迹、RL 训练、评测都用它执行模型发出的工具调用。

规则和线上 bluemed.agents.expert_agent.tools_node 一致：
  - 最多执行 MAX_TOOL_CALLS 次工具，之后的调用只返回 LIMIT_REACHED
  - 和之前某一轮（同工具 + 同参数）重复的调用返回 DUPLICATE_CALL；同一轮里的重复不拦（线上也不拦）
  - 工具在内存图谱上执行（rl/kg_memory.py），返回的文字和证据与线上相同
比线上多处理两种情况（线上会直接抛异常，DeepSeek 几乎不会触发，但小模型训练时很常见）：
  - 工具名不存在、参数不符合 schema（如编造关系类型 REQUIRES_DRUG）→ 返回错误提示，记一次非法调用，也占用次数额度
不提供 analyze_image：影像工具太慢，训练只用纯文本题（RL_PLAN.md 第 2.4 节）。
"""

import json
import uuid
from dataclasses import dataclass, field

from pydantic import ValidationError

from bluemed.agents.expert_agent import DUPLICATE_CALL, LIMIT_REACHED, MAX_TOOL_CALLS
from bluemed.schemas import Evidence
from bluemed.tools.kg_tools import KG_TOOLS
from rl.kg_memory import use_memory_graph

TOOLS_BY_NAME = {t.name: t for t in KG_TOOLS}


@dataclass
class ToolResult:
    """一次工具调用的结果。content 是给模型看的文字（作为 tool 消息接回对话）。"""
    name: str
    args: dict
    content: str
    status: str                                   # ok / duplicate / limit / invalid
    evidence_ids: list[str] = field(default_factory=list)


class ExpertEnv:
    """一道题一个环境。每轮把模型发出的所有工具调用交给 step()，拿回结果接到对话里。"""

    def __init__(self, max_calls: int = MAX_TOOL_CALLS):
        use_memory_graph()
        self.max_calls = max_calls
        self.calls = 0                            # 占用额度的次数（正常执行 + 非法调用）
        self.invalid = 0                          # 非法调用次数（奖励里要扣分）
        self.evidence: dict[str, Evidence] = {}   # 整条轨迹查到的所有证据，按编号合并（同 state["evidence"]）
        self.history: list[ToolResult] = []
        self._done: set[tuple[str, str]] = set()

    def step(self, tool_calls: list[dict]) -> list[ToolResult]:
        """tool_calls：[{"name": ..., "args": {...}}]，即模型这一轮发出的调用（已解析好）。"""
        done_before = set(self._done)             # 线上的重复检测只看之前的轮次
        results = []
        for call in tool_calls:
            name, args = call.get("name", ""), call.get("args") or {}
            args_key = json.dumps(args, ensure_ascii=False, sort_keys=True)
            if self.calls >= self.max_calls:
                result = ToolResult(name, args, LIMIT_REACHED, "limit")
            elif (name, args_key) in done_before:
                result = ToolResult(name, args, DUPLICATE_CALL, "duplicate")
            else:
                result = self._execute(name, args)
                self.calls += 1
            self._done.add((name, args_key))
            self.history.append(result)
            results.append(result)
        return results

    def _execute(self, name: str, args: dict) -> ToolResult:
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            self.invalid += 1
            return ToolResult(name, args, f"没有名为 {name} 的工具。可用工具：{'、'.join(TOOLS_BY_NAME)}。", "invalid")
        try:
            msg = tool.invoke({"type": "tool_call", "id": uuid.uuid4().hex, "name": name, "args": args})
        except ValidationError as e:
            self.invalid += 1
            errors = "；".join(f"{'.'.join(map(str, err['loc']))}：{err['msg']}" for err in e.errors()[:3])
            return ToolResult(name, args, f"参数不合法（{errors}），请按工具说明重新调用。", "invalid")
        artifact = msg.artifact if isinstance(msg.artifact, list) else []
        for e in artifact:
            self.evidence[e.id] = e
        return ToolResult(name, args, msg.content, "ok", [e.id for e in artifact])

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.max_calls
