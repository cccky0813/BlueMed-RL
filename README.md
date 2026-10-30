# BlueMedical v2 · 多 Agent 医疗问答

A multi-modal, multi-agent medical QA system built with **LangGraph** on a Chinese medical knowledge graph (Neo4j).
Three agents collaborate through a shared state: **Question Parser** → **Expert (ReAct + KG / image tools)** → **Answer Generator + Reviewer** (with feedback loops).
Answers follow the user's language (Chinese / English); retrieval always runs in Chinese.

基于 LangGraph 的多模态多 Agent 医疗问答系统：Parser 理解问题并做指代消解 → Expert 用 ReAct 循环调用图谱 / 影像工具取证 → Generator 生成带证据编号的回答 → Reviewer 审核，不合格就打回。

```
START → parser ─拦截（急症/越界/注入）──────────────────────────▶ remember → END
          └─医疗问题─▶ expert ⇄ tools → generator → reviewer ─pass──▶ remember
                         ▲                  ▲          ├─rewrite──────▶ generator
                         └──────────────────┼──────────┴─need_evidence▶ expert
```

## 运行环境

| 依赖 | 说明 |
|---|---|
| Python 3.11 | `conda create -n bluemed python=3.11` |
| Neo4j 5.x | 医疗知识图谱（由 QASystemOnMedicalKG 的 `medical.json` 导入，约 8,800 种疾病）；需要单独导入 |
| DeepSeek API key | Agent 决策和回答生成（OpenAI 兼容接口，其他兼容服务也可以） |
| 可选：本地 LoRA 解析服务 | Qwen2.5-1.5B + LoRA，`mlx_lm server` 跑在 `:8080`，只能在 Apple 芯片上运行；不启动也能用 |
| 可选：FastSAM 权重 | 放到 `weights/FastSAM_X.pt`，用于图片分析 |

## 快速开始

```bash
conda activate bluemed
pip install -r requirements.txt
cp .env.example .env          # 填写 LLM_API_KEY 和 Neo4j 连接信息
python app.py                 # 打开 http://127.0.0.1:5002
```

命令行：

```bash
python scripts/run_pipeline.py "痛风能喝啤酒吗？"    # 打印每个 Agent 的动作
python scripts/chat_demo.py -i                     # 多轮对话
pytest -q                                          # 39 个测试（部分需要 Neo4j）
```

## 目录

```
bluemed/
  agents/      parser_agent · expert_agent · answer_generator · answer_reviewer
  tools/       kg_tools（5 个图谱工具）· image_tools（FastSAM + 视觉模型）· lora_parser
  guardrails/  输入规则（急症 / 注入）
  memory/      健康档案 · SQLite 持久化（短期 checkpoint + 长期 store）
  graph.py     组装 LangGraph 状态图
app.py         Flask + SSE，网页右侧实时显示 Agent 轨迹
tests/         确定性测试（不调大模型）
third_party/   FastSAM 与旧版 ultralytics（AGPL-3.0，来自上游项目）
```

设计要点：手写 ReAct（工具调用上限 + 重复检测）、证据用内容哈希编号以校验引用、Reviewer 规则 + 模型两层审核、短期 / 长期 / 工作记忆三分。
