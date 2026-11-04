"""RL 版 Expert 的系统提示：和线上 EXPERT_PROMPT 只差最后一步的输出格式（RL_PLAN.md 第 2.3 节）。

线上最后输出"证据充分：……/证据不足：……"一句话，程序没法核对；这里改成一行 JSON，奖励函数（rl/reward.py）据此判分。
"""

from bluemed.agents.expert_agent import EXPERT_PROMPT

ONLINE_FINAL = """6. 证据足够时停止调用工具。最后只输出一行，不要写回答（回答由 Answer Agent 负责）：
   "证据充分：<一句话说明找到了什么>" 或 "证据不足：<缺什么>"。"""

RL_FINAL = """6. 证据足够时停止调用工具。最后只输出一行 JSON，不要写回答（回答由 Answer Agent 负责）：
   能回答：{"status": "sufficient", "answer": ["<实体1>", "<实体2>", ...]}
   查不到：{"status": "insufficient", "missing": "<缺什么>"}
   answer 只写能回答问题的实体，名字必须和工具返回里的写法完全一致；问病因、预防等属性时 answer 写 []。
   问"能不能吃 Y"时：查到 Y 在宜吃或忌吃里，answer 写 ["Y"]；宜吃、忌吃里都没有 Y，就是查不到。"""

assert ONLINE_FINAL in EXPERT_PROMPT, "线上 EXPERT_PROMPT 的第 6 条改了，这里要同步"
EXPERT_PROMPT_RL = EXPERT_PROMPT.replace(ONLINE_FINAL, RL_FINAL)
