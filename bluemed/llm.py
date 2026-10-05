"""大模型客户端：所有 Agent 通过 get_llm() 拿到同一种接口的模型。"""

from langchain_openai import ChatOpenAI

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
