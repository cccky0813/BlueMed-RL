"""做题循环：让一个模型完整做一道题（多轮工具调用 → 最后一行 JSON），返回轨迹和奖励。

评测、采老师轨迹、GRPO 用的都是这个循环，只是换模型：
    policy(messages, tools) → {"content": 文字, "tool_calls": [{"name", "args"}...]}
messages 统一用 OpenAI 格式的 dict（system / user / assistant / tool），存下来就是 SFT 能直接用的轨迹。

规则和线上 Expert 一致：工具调用最多 MAX_TOOL_CALLS 次，用完后再生成一次且不再提供工具，逼模型结束
（线上 expert_node 在 tool_calls >= 上限时用不带工具的 llm）。
"""

import json
import re
import time
import uuid
from dataclasses import dataclass, field

from langchain_core.utils.function_calling import convert_to_openai_tool

from bluemed.tools.kg_tools import KG_TOOLS
from rl.env import ExpertEnv
from rl.prompt import EXPERT_PROMPT_RL
from rl.reward import Reward, score

TOOLS = [convert_to_openai_tool(t) for t in KG_TOOLS]
MAX_TURNS = 8                       # 生成轮数上限（防止模型一直发重复调用、不消耗额度地打转）


@dataclass
class Trajectory:
    qid: str
    messages: list[dict]
    final: str | None               # 最后一条不带工具调用的回复；没有就是 None
    reward: Reward
    turns: int
    seconds: float
    tool_status: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"qid": self.qid, "messages": self.messages, "final": self.final, "reward": self.reward.to_dict(),
                "turns": self.turns, "seconds": round(self.seconds, 2), "tool_status": self.tool_status}


def rollout(q: dict, policy, max_turns: int = MAX_TURNS) -> Trajectory:
    env = ExpertEnv()
    if hasattr(policy, "reset"):
        policy.reset()
    messages = [{"role": "system", "content": EXPERT_PROMPT_RL}, {"role": "user", "content": q["question"]}]
    final, turns, t0 = None, 0, time.perf_counter()
    while turns < max_turns:
        turns += 1
        reply = policy(messages, None if env.exhausted else TOOLS)
        calls = reply["tool_calls"] if not env.exhausted else []
        if not calls:
            messages.append({"role": "assistant", "content": reply["content"]})
            final = reply["content"]
            break
        for c in calls:
            c.setdefault("id", "call_" + uuid.uuid4().hex[:8])
        messages.append({"role": "assistant", "content": reply["content"], "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}}
            for c in calls]})
        for c, r in zip(calls, env.step(calls)):
            messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": r.content})
    return Trajectory(q["id"], messages, final, score(q, final, env), turns, time.perf_counter() - t0,
                      [r.status for r in env.history])


# ---------------------------------------------------------------- 本地模型（mlx，Qwen 原生 <tool_call> 格式）
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)


def parse_qwen(text: str) -> dict:
    """把 Qwen 的原始输出拆成文字和工具调用。解析规则和 mlx_lm server 一样严格：不是合法 JSON 的调用丢掉。"""
    calls = []
    for block in TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(block.strip())
            if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                args = obj.get("arguments") or {}
                calls.append({"name": obj["name"], "args": args if isinstance(args, dict) else {}})
        except json.JSONDecodeError:
            pass
    return {"content": TOOL_CALL_RE.sub("", text).strip(), "tool_calls": calls, "raw": text}


class MLXPolicy:
    def __init__(self, model_id: str, temperature: float = 0.0, max_tokens: int = 512):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        self.model_id = model_id
        self.model, self.tok = load(model_id)
        self.sampler = make_sampler(temp=temperature)
        self.max_tokens = max_tokens

    def __call__(self, messages: list[dict], tools: list[dict] | None) -> dict:
        from mlx_lm import generate
        prompt = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        text = generate(self.model, self.tok, prompt=prompt, max_tokens=self.max_tokens,
                        sampler=self.sampler, verbose=False)
        return parse_qwen(text)


class VLLMPolicy:
    """服务器上的本地模型：vLLM 推理，可挂 SFT / GRPO 训出的 LoRA。模板和解析与 MLXPolicy 相同，结果可直接对比。"""
    def __init__(self, model_path: str, temperature: float = 0.0, max_tokens: int = 512, lora: str | None = None):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        self.model_id = model_path + (f" + {lora}" if lora else "")
        self.llm = LLM(model_path, enable_lora=bool(lora), max_lora_rank=64, max_model_len=8192,
                       gpu_memory_utilization=0.85, seed=0)
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        self.lora = LoRARequest("adapter", 1, lora) if lora else None

    def __call__(self, messages: list[dict], tools: list[dict] | None) -> dict:
        prompt = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        out = self.llm.generate([prompt], self.params, lora_request=self.lora, use_tqdm=False)
        return parse_qwen(out[0].outputs[0].text)


# ---------------------------------------------------------------- DeepSeek（线上 Expert 用的模型，原生 function calling）
class DeepSeekPolicy:
    def __init__(self, temperature: float = 0.0):
        from bluemed.llm import get_llm
        self.llm = get_llm(temperature=temperature)          # 和线上 Expert 一样：保留思考
        self.model_id = "deepseek (" + self.llm.model_name + ")"
        self._raw: dict[int, object] = {}                    # 思考模式要把原始 AIMessage 原样传回去

    def __call__(self, messages: list[dict], tools: list[dict] | None) -> dict:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        lc = []
        for i, m in enumerate(messages):
            if m["role"] == "system":
                lc.append(SystemMessage(m["content"]))
            elif m["role"] == "user":
                lc.append(HumanMessage(m["content"]))
            elif m["role"] == "tool":
                lc.append(ToolMessage(m["content"], tool_call_id=m["tool_call_id"]))
            else:
                lc.append(self._raw.get(i) or AIMessage(m["content"]))
        llm = self.llm.bind_tools(KG_TOOLS) if tools else self.llm
        msg = llm.invoke(lc)
        self._raw[len(messages)] = msg
        calls = [{"name": c["name"], "args": c["args"], "id": c["id"]} for c in msg.tool_calls]
        return {"content": msg.content if isinstance(msg.content, str) else "", "tool_calls": calls}

    def reset(self) -> None:
        self._raw.clear()
