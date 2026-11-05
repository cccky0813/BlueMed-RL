#!/usr/bin/env bash
# 难测评集（rl/hard_set.py）：原版和 A 组各测一遍；B / C 训完后加上它们的 LoRA 再跑。用法同 eval_sft.sh。
set -uo pipefail
cd /root/autodl-tmp/BlueMed-RL
export HF_HOME=/root/autodl-tmp/.cache/huggingface HF_HUB_OFFLINE=1
PY=/root/autodl-tmp/venv/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-2507
for split in hard_free hard_lay hard_distractor; do
  $PY -m rl.eval --model $MODEL --engine vllm --split $split
  for lora in "$@"; do $PY -m rl.eval --model $MODEL --engine vllm --split $split --lora $lora; done
done
echo "EVAL HARD DONE"
