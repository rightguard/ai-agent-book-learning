# retrieval-pipeline — 五阶段检索流水线（实验 3-6）

纯 DashScope API 的混合检索评测——没有本地模型、没有检索微服务、没有仓库内 Python 导入。**同一个 20 文档 / 12 查询小型标注语料**，依次跑 BM25、Dense、Hybrid-RRF、Hybrid-Weighted、Hybrid-RRF+Rerank 五个阶段，看每一步各指标（Recall@3 / MRR / nDCG / latency）的变化。

## 文件

| 文件 | 说明 |
|------|------|
| `experiment_3_6.py` | 主脚本：内嵌语料 + 内嵌查询集，5 个阶段串行跑，一键出汇总 |
| `results/exp3_6_20261015_122245_795517/` | `summary.md`（Markdown 结果表）+ `results.json`（完整轨迹）+ `per_query.csv`（逐题）+ `category_summary.csv`（XR/HTTP/语义/姓名/长文档分组） |

## 运行

```bash
# 必须从仓库根目录启动，读取根 config.yaml + .env
python char3/retrieval-pipeline/experiment_3_6.py

# 自测试（不消耗 embedding / rerank API，用模拟 HTTP 响应）
python char3/retrieval-pipeline/experiment_3_6.py --self-test

# 只跑 BM25 baseline
python char3/retrieval-pipeline/experiment_3_6.py --no-dense
```

## 五个阶段

| 阶段 | 做法 | 为什么有这一步 |
|------|------|----------------|
| **BM25** | 纯词频匹配 | 基线——传统检索 |
| **Dense** | qwen3.7-text-embedding-flash API 做向量检索 | 语义匹配的上限 |
| **Hybrid-RRF** | BM25 + Dense 用 Reciprocal Rank Fusion 融合 | 弥补 BM25 语义盲区、弥补 Dense 精确词盲区 |
| **Hybrid-Weighted** | BM25 + Dense 加权线性融合 | 探索比 RRF 更灵活的融合策略 |
| **Hybrid-RRF+Rerank** | RRF 候选集 → qwen3.7-text-rerank API 重排 | 最后一公里精排 |

## 2026-10-15 运行结果摘要

| Stage | Recall@3 | Hit@3 | MRR | nDCG@3 | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.8333 | 0.8333 | 0.7917 | 0.8026 | 113.8 |
| **Dense** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 346.4 |
| Hybrid-RRF | 1.0000 | 1.0000 | 0.9583 | 0.9692 | 429.6 |
| Hybrid-Weighted | 1.0000 | 1.0000 | 0.9583 | 0.9692 | 429.6 |
| **Hybrid-RRF+Rerank** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 801.4 |

**关键发现**：

- **Dense embedding 在这个 20 文档小型 benchmark 上就已经拿到了完美 MRR/nDCG**——Hybrid 和 Rerank 的收益在 top-3 指标里没体现出来，因为 RRF 候选集本身就没漏，Rerank 也不可能"召回新文档"；
- RRF 和 Weighted 融合的 Recall@k 也都是完美，但 MRR 降到 0.96——说明融合后第一名不一定是最优，这是因为 BM25 在某些题目上排到了前面；
- 延迟从 BM25 的 114ms → Dense 的 346ms → RRF 的 430ms → +Rerank 的 801ms；融合和精排的成本是每请求 450ms 额外开销；
- **这是 20 文档 12 查询的小型教育 benchmark**，不能外推到真实大规模语料；真实语料里 Dense 很可能会漏，RRF+Rerank 就有用了。

完整轨迹和 API 回执见 `results/exp3_6_20261015_122245_795517/`。
