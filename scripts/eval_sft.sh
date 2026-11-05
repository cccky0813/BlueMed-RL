#!/usr/bin/env bash
# 等 SFT 训完，在 dev（模板 + 口语）上测原版 4B 和 A 组，同一个推理引擎（vLLM bf16，temperature 0）。
# 用法（服务器上）：nohup bash scripts/eval_sft.sh > ../logs/eval_sft.log 2>&1 &
set -uo pipefail
cd /root/autodl-tmp/BlueMed-RL
export HF_HOME=/root/autodl-tmp/.cache/huggingface HF_HUB_OFFLINE=1
PY=/root/autodl-tmp/venv/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-2507
LORA=/root/autodl-tmp/ckpt/sft/final

until [ -f $LORA/adapter_model.safetensors ]; do sleep 30; done
sleep 30                                             # 等训练进程退出、释放显存

for split in dev dev_oral; do
  $PY -m rl.eval --model $MODEL --engine vllm --split $split                 # 基线：原版
  $PY -m rl.eval --model $MODEL --engine vllm --split $split --lora $LORA    # A 组：只 SFT
done
echo "EVAL DONE"
