"""大模型客户端：所有 Agent 通过 get_llm() 拿到同一种接口的模型。"""

import json

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from bluemed import config


def get_llm(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    """返回一个 OpenAI 兼容接口的聊天模型。换模型 / 换服务商只需要改 .env。"""
    return ChatOpenAI(
        model=model or config.LLM_MODEL,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )


def structured_invoke(schema: type[BaseModel], prompt: str, retries: int = 1):
    """让模型按 Pydantic schema 输出，返回校验过的对象。

    结构化输出有三种实现，保证程度从强到弱：
      json_schema    服务端按 schema 约束解码（DeepSeek 不支持）
      function_calling 把 schema 当作工具强制模型调用（deepseek-flash 的思考模式不支持强制 tool_choice）
      json_mode      只保证输出是合法 JSON（DeepSeek 支持）← 我们用这个
    json_mode 不保证字段正确，所以自己补两道保障：
      1. 把 schema（字段、类型、说明、可选值）写进 prompt
      2. 用 Pydantic 校验，失败就把错误信息发回给模型重新输出（最多 retries 次）
    """
    fields = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    messages = [("user", f"{prompt}\n\n只输出一个 JSON 对象，必须符合下面的 JSON Schema：\n{fields}")]
    llm = get_llm().bind(response_format={"type": "json_object"})
    for attempt in range(retries + 1):
        reply = llm.invoke(messages)
        try:
            return schema.model_validate_json(reply.content)
        except ValidationError as e:
            if attempt == retries:
                raise
            messages += [("assistant", reply.content),
                         ("user", f"输出不符合 schema：{e.errors()[:3]}。请修正后只输出 JSON。")]
