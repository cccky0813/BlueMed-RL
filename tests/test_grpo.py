"""GRPO 的逻辑测试（Mac 上跑，不需要 GPU 和 vLLM）：
假生成器测批量多轮采样；随机初始化的小 Qwen3 测 loss 的遮罩和梯度方向。运行：python -m pytest tests/test_grpo.py -q"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.grpo import group_advantages, policy_loss, rl_pool, rollout_batch
from rl.make_questions import build
from rl.reward import FORMAT_FAIL
from rl.rollout import TOOLS
from rl.sft_data import is_sft
from rl.train_sft import encode, render


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("mlx-community/Qwen3-4B-Instruct-2507-4bit",
                                                          local_files_only=True)
    except OSError:
        pytest.skip("本地没有 Qwen3 分词器")


@pytest.fixture(scope="module")
def dev():
    return build(seed=0)["dev"]


def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"


def test_pool_excludes_sft_questions():
    pool = rl_pool()
    assert all(not is_sft(q["id"]) for q in pool)
    assert 5800 < len(pool) < 6300
    share = sum(q["id"].endswith("-oral") for q in pool) / len(pool)
    assert 0.15 < share < 0.35                        # 一半有口语版，其中一半换成口语 → 约 1/4


def test_rollout_batch_multi_turn(dev, tok):
    fwd = next(q for q in dev if q["type"] == "forward" and len(q["relations"]) == 1 and q["subtype"] != "food_pair")
    seen = []

    def gen(prompts):
        seen.append(len(prompts))
        out = []
        for p in prompts:
            if "<tool_response>" not in p:                 # 第一轮：查一次
                out.append(call("kg_search", entity=fwd["anchor"], relation=fwd["relations"][0]))
            elif fwd["question"] in p:                      # 第二轮：交答案
                out.append(json.dumps({"status": "sufficient", "answer": fwd["gold"]["answer"]}, ensure_ascii=False))
            else:
                out.append("我不知道")
        return out

    other = dict(fwd, id="x", question="随便问问")
    trajs = rollout_batch([fwd, fwd, other], gen, tok)
    assert seen == [3, 3]                                  # 三条一起采，第二轮三条都还在
    assert [t.turns for t in trajs] == [2, 2, 2]
    assert trajs[0].reward.total == 1.0 and trajs[1].reward.total == 1.0
    assert trajs[2].reward.total == FORMAT_FAIL            # 最后一句不是 JSON
    assert trajs[0].messages[2]["content"].startswith("<tool_call>")    # 存的是原始输出


def test_generation_prompts_are_prefixes_of_training_text(dev, tok):
    """每一轮采样时看到的提示，必须是训练时渲染整条轨迹的前缀，否则算概率的 token 和生成的 token 对不上。"""
    fwd = next(q for q in dev if q["type"] == "forward" and q["subtype"] != "food_pair")
    prompts = []

    def gen(ps):
        prompts.extend(ps)
        return [call("kg_search", entity=fwd["anchor"], relation=fwd["relations"][0]) if len(prompts) == 1
                else '{"status": "insufficient", "missing": "x"}' for _ in ps]

    t = rollout_batch([fwd], gen, tok)[0]
    full = render(tok, {"messages": t.messages, "tools": TOOLS})
    for p in prompts:
        assert full.startswith(p)
    e = encode(tok, {"messages": t.messages, "tools": TOOLS}, 10 ** 6)
    learned = tok.decode([i for i, l in zip(e["input_ids"], e["labels"]) if l != -100])
    assert learned.count("<|im_end|>") == 2 and "<tool_response>" not in learned


def test_group_advantages():
    assert group_advantages([1.0] * 8) is None             # 全对：没信号
    assert group_advantages([0.0] * 8) is None
    a = group_advantages([1.0, 0.0, 0.0, 1.0])
    assert a == [1.0, -1.0, -1.0, 1.0]


def test_policy_loss_masks_and_direction(dev, tok):
    """随机小模型：A > 0 走一步后，要学的 token 概率上升；A = 0 时 LoRA 没有梯度；系统提示等位置不影响 loss。"""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=8192)
    model = peft.get_peft_model(Qwen3ForCausalLM(cfg), peft.LoraConfig(r=4, lora_alpha=8, target_modules="all-linear",
                                                                       init_lora_weights=False))
    fwd = next(q for q in dev if q["type"] == "forward")
    msgs = [{"role": "system", "content": "系统"}, {"role": "user", "content": fwd["question"]},
            {"role": "assistant", "content": '{"status": "insufficient", "missing": "x"}'}]
    e = encode(tok, {"messages": msgs, "tools": TOOLS}, 10 ** 6)
    n = sum(l != -100 for l in e["labels"])

    loss, st = policy_loss(model, [e], [0.0], n, tok.pad_token_id)
    loss.backward()
    assert st["tokens"] == n
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in model.parameters() if p.requires_grad)

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    before = policy_loss(model, [e], [1.0], n, tok.pad_token_id)[1]["logp_sum"]
    for _ in range(3):
        opt.zero_grad()
        policy_loss(model, [e], [1.0], n, tok.pad_token_id)[0].backward()
        opt.step()
    after = policy_loss(model, [e], [1.0], n, tok.pad_token_id)[1]["logp_sum"]
    assert after > before
