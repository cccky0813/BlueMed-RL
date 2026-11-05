#!/usr/bin/env bash
# GRPO（RL_PLAN.md 第 6.2 节）。用法（服务器上，nohup 挂后台）：
#   bash scripts/run_grpo.sh smoke     # 冒烟：4 组 × 8 条、3 步，测速度和显存（约 10 分钟）
#   bash scripts/run_grpo.sh b         # B 组：原版 → 筛题 → 300 步 → 测评
#   bash scripts/run_grpo.sh c         # C 组：SFT → 筛题 → 120 步 → 测评
#   bash scripts/run_grpo.sh bc shutdown   # 无人值守：B、C 依次跑完后自动关机（出错也会关）
#   加第二个参数 shutdown：跑完自动关机（省钱规则第 3 条）
set -uo pipefail
cd /root/autodl-tmp/BlueMed-RL
export HF_HOME=/root/autodl-tmp/.cache/huggingface HF_HUB_OFFLINE=1
PY=/root/autodl-tmp/venv/bin/python
MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-2507
CK=/root/autodl-tmp/ckpt
SFT=$CK/sft/final

evaluate() {   # $1 = LoRA 目录
  for split in dev dev_oral test hard_free hard_lay hard_distractor; do
    $PY -m rl.eval --model $MODEL --engine vllm --split $split --lora $1
  done
}

case "$1" in
  smoke)
    $PY -m rl.grpo train --init base --out $CK/grpo_smoke --steps 3 --n-groups 4 --eval-every 3 --save-every 3 ;;
  b)
    [ -f $CK/grpo_b/pool.jsonl ] || $PY -m rl.grpo prefilter --init base --out $CK/grpo_b
    $PY -m rl.grpo train --init base --out $CK/grpo_b --steps 300
    evaluate $CK/grpo_b/best ;;            # dev 上最好的那个 LoRA（可能提前停）
  c)
    [ -f $CK/grpo_c/pool.jsonl ] || $PY -m rl.grpo prefilter --init $SFT --out $CK/grpo_c
    $PY -m rl.grpo train --init $SFT --out $CK/grpo_c --steps 120 --min-steps 60
    evaluate $CK/grpo_c/best ;;
  bc)
    bash scripts/eval_hard.sh $SFT          # 难测评集上的原版和 A 组
    bash $0 b; bash $0 c ;;
esac
echo "RUN $1 DONE"
[ "${2:-}" = "shutdown" ] && shutdown
