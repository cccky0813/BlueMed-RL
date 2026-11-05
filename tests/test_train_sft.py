"""SFT 遮罩测试：只学 assistant 段，去掉模板插的空 think。需要本地缓存里有 Qwen3 的分词器，没有就跳过。
运行：python -m pytest tests/test_train_sft.py -q"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.rollout import TOOLS
from rl.train_sft import encode, render


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("mlx-community/Qwen3-4B-Instruct-2507-4bit",
                                                          local_files_only=True)
    except OSError:
        pytest.skip("本地没有 Qwen3 分词器")


EX = {"tools": TOOLS, "messages": [
    {"role": "system", "content": "系统提示"},
    {"role": "user", "content": "痛风不能吃什么？"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "kg_search", "arguments": '{"entity": "痛风", "relation": "SHOULD_NOT_EAT"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "kg_search", "content": "[E1] 痛风 -SHOULD_NOT_EAT-> 啤酒"},
    {"role": "assistant", "content": '{"status": "sufficient", "answer": ["啤酒"]}'},
]}


def test_only_assistant_tokens_are_learned(tok):
    e = encode(tok, EX, 10 ** 6)
    learned = tok.decode([t for t, l in zip(e["input_ids"], e["labels"]) if l != -100])
    assert e["n_spans"] == 2
    assert '"name": "kg_search"' in learned and '"answer": ["啤酒"]' in learned
    for hidden in ["系统提示", "痛风不能吃什么", "痛风 -SHOULD_NOT_EAT-> 啤酒", "<tool_response>"]:
        assert hidden not in learned
    assert learned.count("<|im_end|>") == 2


def test_empty_think_removed_and_prefix_matches_generation_prompt(tok):
    text = render(tok, EX)
    assert "<think>" not in text
    gen = tok.apply_chat_template(EX["messages"][:2], tools=TOOLS, add_generation_prompt=True, tokenize=False)
    assert text.startswith(gen)
