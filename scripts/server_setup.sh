#!/usr/bin/env bash
# AutoDL 服务器环境（RL_PLAN.md 第 6 节）：在数据盘上建 venv，复用镜像自带的 torch，装训练和项目依赖，下载模型。
# 用法（在服务器上）：bash /root/autodl-tmp/BlueMed-RL/scripts/server_setup.sh
# 11/04 首次在 RTX 6000D（Blackwell，sm_120）+ 镜像 PyTorch 2.8.0 / CUDA 12.8 上跑通。
set -euo pipefail

ROOT=/root/autodl-tmp
export PIP_CACHE_DIR=$ROOT/.cache/pip HF_HOME=$ROOT/.cache/huggingface MODELSCOPE_CACHE=$ROOT/.cache/modelscope
export HF_ENDPOINT=https://hf-mirror.com              # m3e-base 从 HF 镜像下

# 1. venv 放数据盘（系统盘只有 30GB）；--system-site-packages 复用镜像里的 torch 2.8.0+cu128
[ -d $ROOT/venv ] || /root/miniconda3/bin/python -m venv --system-site-packages $ROOT/venv
PIP=$ROOT/venv/bin/pip
$PIP install -U pip

# 2. 训练：vLLM 0.11.0 依赖的正是 torch 2.8.0，不会重装 torch。
#    transformers 必须 < 5（vLLM 0.11 用到了 5.x 删掉的 all_special_tokens_extended），其余跟着它锁版本（11/04 踩过）
$PIP install "vllm==0.11.0" "transformers==4.57.1" "trl==0.24.0" "datasets==4.3.0" peft accelerate modelscope tensorboard

# 3. 项目本身（rl/env.py 用到 bluemed 的工具、实体链接用 m3e）；影像、Web 相关的依赖不装
$PIP install langgraph==1.2.12 langgraph-checkpoint-sqlite==3.1.1 langchain-core==1.6.6 langchain-openai==1.6.7 \
    pydantic==2.13.5 python-dotenv==1.2.4 neo4j==6.3.1 "sentence-transformers==5.1.2" pytest

# 4. 模型：主模型从 ModelScope 下（国内快），m3e 预先拉到缓存
$ROOT/venv/bin/modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir $ROOT/models/Qwen3-4B-Instruct-2507
# m3e：HF 镜像下载时解压报错（11/04），改成从 Mac 拷：rsync -azL ~/.cache/huggingface/hub/models--moka-ai--m3e-base 服务器:$HF_HOME/hub/
HF_HUB_OFFLINE=1 $ROOT/venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('moka-ai/m3e-base')"

$ROOT/venv/bin/python -c "import torch, vllm, trl, peft, transformers; print('torch', torch.__version__, torch.cuda.get_device_name(0)); print('vllm', vllm.__version__, 'trl', trl.__version__, 'peft', peft.__version__, 'transformers', transformers.__version__)"
echo "SETUP DONE"
