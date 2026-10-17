# contextual-retrieval — Contextual Retrieval 索引增强对照（实验 3-10）

**Contextual Retrieval**（Anthropic 2024 提出的技术）的核心想法：索引时，让 LLM 给每个 chunk 生成一段"这个 chunk 在整篇文档里的位置/角色/上下文摘要"作为前缀，再用**原始 chunk + 生成前缀**做 embedding。对照 six 种方法：plain vs contextual × bm25 / dense / hybrid。

## 文件

| 文件 | 说明 |
|------|------|
| `experiment_3_10.py` | 主脚本：加载官方 pin 的 22-chunk / 15-query 小型法律 fixture，跑 `--method all` 一键出六种对照 |
| `official_reference_chunks.json` | 官方 pin 的 chunk fixture（22 个 chunk，来自宪法 + 检察官法） |
| `official_retrieval_eval_15.json` | 官方 pin 的 15 条标注查询 + gold chunk 映射 |
| `result.md` | 2026-10-16 完整终端运行日志 |
| `cache/embeddings/` | 每个 chunk 的 embedding JSON（命中缓存时不消耗 API） |
| `cache/contexts/` | 每个 chunk 的 contextual prefix JSON（LLM 生成的原文外摘要） |
| `results/exp3_10_20261016_103203_781638/` | `summary.md` / `summary.csv` / `results.json` + 排名变化 `rank_differences.csv` + 前缀审计 `prefix_audit.csv` |

## 运行

```bash
# 从仓库根目录启动
python char3/contextual-retrieval/experiment_3_10.py --mode evaluate --method all --confirm-api-cost

# 只跑 dense vs contextual_dense
python char3/contextual-retrieval/experiment_3_10.py --method dense

# 跳过 prefix 生成（用已缓存的）
python char3/contextual-retrieval/experiment_3_10.py --skip-prefix
```

关键模型参数都有默认值可覆盖：

- 上下文生成模型：`--context-model qwen3.8-27b`（昂贵但质量好，也可以换成 flash 版本）
- Embedding 模型：默认 DashScope `qwen3.7-text-embedding-flash`（1024 维）

## 六种对照方法

| 方法 | 做法 |
|------|------|
| plain_bm25 | 纯原始 chunk 文本做 BM25 匹配（基线） |
| contextual_bm25 | 原始 chunk + LLM 生成前缀拼起来做 BM25 |
| plain_dense | 纯原始 chunk 做 embedding → dense 检索 |
| contextual_dense | **原始 chunk + 前缀拼起来做 embedding** |
| plain_hybrid | plain_bm25 + plain_dense → RRF 融合 |
| contextual_hybrid | contextual_bm25 + contextual_dense → RRF 融合 |

## 2026-10-16 运行结果摘要

数据：2 文档（宪法 + 检察官法）→ 22 chunks → 15 查询

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| plain_bm25 | 60.00% | 86.67% | 93.33% | 0.7508 |
| **contextual_bm25** | **73.33%** | 86.67% | 93.33% | **0.8139** |
| plain_dense | 86.67% | 100.00% | 100.00% | 0.9333 |
| **contextual_dense** | **93.33%** | 100.00% | 100.00% | **0.9667** |
| plain_hybrid | 66.67% | 93.33% | 100.00% | 0.8133 |
| **contextual_hybrid** | **73.33%** | 93.33% | 100.00% | **0.8500** |

**逐题排名变化**：

- BM25：改善 4 题（q01/q02/q08/q09），退化 1 题（q15）
- Dense：改善 1 题（q02），**退化 0 题**
- Hybrid：改善 3 题（q02/q08/q09），退化 1 题（q15）

**关键发现**：

- **Recall@k / MRR 的改进几乎全部集中在 Top-1**——Recall@3 和 Recall@5 三种检索方式都已经接近满分，加 prefix 在这些高位指标上榨不出更多；
- **Dense 检索 + Contextual prefix 是 15 题里唯一"零退化"的改进**——Hybrid 和 BM25 都至少有一题被带偏（q15），因为 prefix 可能引入原文里没有的事实；
- **索引期成本 ≠ 查询期成本**：prefix 生成 + 文档 embedding 是一次性索引开销（本实验 22 chunk × qwen3.8-27b ≈ 22 次大模型调用），查询期只有 query embedding；真实生产里索引成本可摊薄到"一次建索引多次查询"；
- **15 题每题只有一个 gold chunk**——Recall@k 数值等价于 Hit@k；真实评估需要多 gold 题和更多 query 类型。

完整排名变化、前缀审计、API 消耗全在 `results/exp3_10_20261016_103203_781638/` 下。
