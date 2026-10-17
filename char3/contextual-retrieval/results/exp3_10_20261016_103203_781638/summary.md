# 实验 3-10：Contextual Retrieval 索引增强对照

数据来源：LIVE_QWEN_PREFIX_GENERATION_FULL
Chunk：22；查询数：15
上下文生成模型：qwen3.8-27b
Embedding 模型：qwen3.7-text-embedding-flash

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| plain_bm25 | 60.00% | 86.67% | 93.33% | 0.7508 |
| contextual_bm25 | 73.33% | 86.67% | 93.33% | 0.8139 |
| plain_dense | 86.67% | 100.00% | 100.00% | 0.9333 |
| contextual_dense | 93.33% | 100.00% | 100.00% | 0.9667 |
| plain_hybrid | 66.67% | 93.33% | 100.00% | 0.8133 |
| contextual_hybrid | 73.33% | 93.33% | 100.00% | 0.8500 |

重要说明：15 道题每道只有一个 Gold Chunk，因此此处 Recall@k 数值等价于 Hit@k。
离线公开快照不能代表本地 Qwen 生成前缀的效果；实际完整比较应运行 evaluate 模式。
改进可能仅发生在 Top-1；必须逐题检查损害案例以及生成前缀是否包含未经原文支持的内容。
Index-time Chat/Embedding 开销与 query-time Embedding/Rerank 开销分开记录。
本示例为法律文档检索教学评测；语料并非现行法律有效性验证。

## 逐题排名变化（增强相对于普通分块）

- bm25: 排名改善 4 题（q01, q02, q08, q09）；排名退化 1 题（q15）
- dense: 排名改善 1 题（q02）；排名退化 0 题（无）
- hybrid: 排名改善 3 题（q02, q08, q09）；排名退化 1 题（q15）

## 调用成本 / 原文可核验性

- 上下文前缀生成和文档 Embedding 是索引期成本，query Embedding / Rerank 是查询期成本。
- 逐阶段 API 消耗：`api_receipts.jsonl` / `results.json` 的 `api.stage_times` 与 `api.usage`。
- 请打开 `prefix_audit.csv` 将新前缀逐块比对原文，拒绝引入原文不支持的事实。
- 仅使用同一组 Chunk ID / Gold ID 进行有效对照；其他分块规则需要重新标注。
