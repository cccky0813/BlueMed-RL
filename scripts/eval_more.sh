#!/usr/bin/env bash
# A 组之后的补充测评：pass@8（GRPO 还有多少可学）和 test 集。用法同 eval_sft.sh。
set -uo pipefail
cd /root/autodl-tmp/BlueMed-RL
export HF_HOME=/root/autodl-tmp/.cache/huggingface HF_HUB_OFFLINE=1
PY=/root/autodl-tmp/venv/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-2507
LORA=/root/autodl-tmp/ckpt/sft/final
P8="--per-type 10 --samples 8 --temperature 1.0"

$PY -m rl.eval --model $MODEL --engine vllm --split dev $P8 --lora $LORA     # C 组起点的 pass@8
$PY -m rl.eval --model $MODEL --engine vllm --split dev $P8                  # B 组起点（bf16，和 Mac 4bit 对照）
$PY -m rl.eval --model $MODEL --engine vllm --split test
$PY -m rl.eval --model $MODEL --engine vllm --split test --lora $LORA
echo "EVAL MORE DONE"
