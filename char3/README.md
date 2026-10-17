# Chapter 3 — RAG、Agentic Retrieval 与索引增强

本章三个实验覆盖 **检索流水线（BM25 / Dense / Hybrid / Rerank）、Agentic RAG（ReAct vs One-shot）、Contextual Retrieval（索引时 LLM 生成上下文前缀）** 三个方向。所有脚本共用仓库根目录的 `config.yaml` + `.env`，**请从仓库根目录启动**。

## 目录一览

| 目录 | 实验 | 主题 |
|------|------|------|
| [retrieval-pipeline/](retrieval-pipeline/README.md) | 3-6 | 五阶段检索流水线评测：BM25 → Dense → Hybrid-RRF → Hybrid-Weighted → RRF+Rerank |
| [agentic-rag/](agentic-rag/README.md) | 3-8 | ReAct Agentic RAG vs 单轮 RAG 对照——中文法律问答，标注 gold 法条 |
| [contextual-retrieval/](contextual-retrieval/README.md) | 3-10 | Contextual Retrieval：索引时 LLM 生成 chunk 前缀，对照 plain vs contextual × bm25/dense/hybrid |
| 心得.md | — | 三个实验的个人总结与工程启示 |
| 思考题.md | — | 章节 6 道思考题（含个人回答，难度 ★★ ~ ★★★） |

**数据 / 缓存目录**（不是实验脚本）：

| 目录 | 说明 |
|------|------|
| `agentic-rag/laws/` | 288 个中文法律 .md 文件（宪法 + 民法典 + 刑法 + 各部门法），实验 3-8 的本地语料库 |
| `contextual-retrieval/cache/embeddings/` | 每个 chunk 的 embedding JSON（命中缓存时不消耗 API） |
| `contextual-retrieval/cache/contexts/` | 每个 chunk 的 contextual prefix JSON（LLM 生成的原文外摘要） |

> 注：实验编号 3-7 / 3-9 在 `char3/` 下没有对应子目录，是教程另一章的实验。

## 环境准备

```bash
pip install -r requirements.txt
```

### API Key + config.yaml

根目录 `.env` 至少需要：

```
# DashScope（实验 3-8 / 3-10 的 LLM 调用，实验 3-6 的 Embedding + Rerank）
DASHSCOPE_API_KEY=你的通义百炼 key
QWEN_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### Jieba（实验 3-8 中文分词）

```bash
pip install jieba
```

实验 3-6 的 Dense / Rerank 和实验 3-10 的 Embedding / Contextual Prefix 都是 DashScope 提供的托管 API（`qwen3.7-text-embedding-flash`、`qwen3.7-text-rerank`），不加载本地模型。

## 快速运行

```bash
# 实验 3-6：五阶段检索流水线
python char3/retrieval-pipeline/experiment_3_6.py
python char3/retrieval-pipeline/experiment_3_6.py --self-test  # 不消耗 API

# 实验 3-8：Agentic vs One-shot RAG
python char3/agentic-rag/experiment_3_8.py --mode evaluate --confirm-api-cost
python char3/agentic-rag/experiment_3_8.py --mode demo           # 小型语料 smoke test

# 实验 3-10：Contextual Retrieval
python char3/contextual-retrieval/experiment_3_10.py --mode evaluate --method all --confirm-api-cost
python char3/contextual-retrieval/experiment_3_10.py --skip-prefix  # 用已缓存的 prefix
```

## 关键结论索引

每个子目录的 README 里有完整结果表和分析，这里按实验编号把**本次（2026-10-15 / 16）运行得到的核心结论**集中列出来：

| 实验 | 问题 | 核心结论 |
|------|------|----------|
| 3-6 | BM25 / Dense / Hybrid / Rerank 每个阶段到底有多少增益 | Dense embedding 在 20 文档小型 benchmark 上**就已经拿到完美 MRR/nDCG**（1.0000）；Hybrid 和 Rerank 没有额外收益——因为候选池本身没漏、精排也排不回"没召回"的文档。真实大规模语料里 Dense 会漏，RRF+Rerank 才有用 |
| 3-8 | ReAct Agent 让模型"自己改 query"值不值 | 在 hard_2 / hard_3 上**Evidence recall 从 0% 直接升到 100%**（模型主动把"盗窃 立案"扩成"盗窃 公私财物 数额较大 有期徒刑"）；但 hard_4 上 agentic 反而掉到 0%——模型"自信地"决定这道题靠自己知识就能答，**一次搜索都没发** |
| 3-10 | 给 chunk 加 LLM 生成的 context 前缀能不能提升检索 | contextual_dense 的 MRR **从 0.93 升到 0.97**，Recall@1 从 86.7% 升到 93.3%；**改进全部集中在 Top-1**，Recall@3/@5 已经接近满分；Dense+contextual 是唯一**零退化**的方案（Hybrid 和 BM25 都有一题被带偏） |

**检索质量 vs LLM 生成质量的 tradeoff**：实验 3-10 的 contextual prefix 是 qwen3.8-27b 生成的——如果生成的前缀里有原文里没有的事实（"幻觉 prefix"），就会把 query 的 embedding 往错误方向带。脚本里 `prefix_audit.csv` 提供了逐块对比原文的钩子。

## 小提示

- **实验 3-6 没有仓库内 import**：独立单文件脚本，所有语料和查询都内嵌在 `CORPUS` / `TASK_SPEC` 常量里，方便 self-contained 运行；
- **实验 3-8 laws/ 目录 288 个 md**：jieba 分词 + 滑动窗口切 chunk 大概 21,085 块，索引阶段 15-20 秒；ReAct 模式下每道题会发多轮 API（easy_1~3 各 1 轮，hard_2~3 各 3 轮）；
- **实验 3-10 的 contextual prefix + embedding 是索引期成本**：一次性可摊薄，不要和 query 期的 embedding/Rerank 开销混为一谈；
- 三个实验的 API 消耗和 token 统计都存在各 results/ 下的 `api_receipts.json(l)` 或 `results.json.api.stage_times` 里，**不包含密钥**。
