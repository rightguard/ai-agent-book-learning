# agentic-rag — ReAct Agentic vs One-shot RAG（实验 3-8）

同一个中文法律问答任务集（7 题 easy + hard，标注了 gold 法条编号），**两条路线对比**：

- **agentic**：ReAct Agent，手里有一个 `search_laws` 工具，模型自主决定搜几次、搜什么关键词、什么时候停；
- **non-agentic**：单轮 RAG，把所有检索结果一次性塞给模型让它出答案，模型不参与检索决策。

## 文件

| 文件 | 说明 |
|------|------|
| `experiment_3_8.py` | 主脚本：内嵌官方标注查询集 + 7 题 gold_articles，`--mode evaluate` 一键跑两种方法 |
| `laws/` | **本地法律语料库**：288 个 .md 法律文件（宪法 + 民法典 + 刑法 + 各部门法），脚本会 jieba 分词、chunk 成 21,085 块 |
| `result.md` | 2026-10-15 完整终端运行日志 |
| `results/exp3_8_20261015_143724_812927/` | `results.json`（完整执行记录）+ `trajectories.jsonl`（ReAct 逐轮轨迹）+ `per_query.csv`（逐题对照）+ `summary.md`（汇总表） |

## 运行

```bash
# 从仓库根目录启动
python char3/agentic-rag/experiment_3_8.py --mode evaluate --confirm-api-cost

# 先只跑 demo 语料 smoke test
python char3/agentic-rag/experiment_3_8.py --mode demo

# 只跑 ReAct 单次轨迹（用于观察推理过程）
python char3/agentic-rag/experiment_3_8.py --mode react
```

脚本用 jieba 对 laws/ 下 288 个法律文件（含宪法、民法典、刑法）做中文分词 + 滑动窗口切 chunk（21,085 块），然后让 qwen3.7-flash 通过 DashScope OpenAI-compatible API 跑 ReAct。

## 2026-10-15 运行结果摘要

| Method | Cases | Evidence recall | Mean searches | Mean latency ms | Valid cite rate |
|---|---:|---:|---:|---:|---:|
| **agentic** | 7 | **71.4%** | 1.29 | 18,862 | 57.1% |
| non-agentic | 7 | 47.6% | 1.00 | 22,324 | 71.4% |

**Evidence recall（证据召回率）是金法条检索命中率**——不是答案正确性。

逐题细看：

| 题目 | 难度 | agentic recall | non-agentic recall | 差异原因 |
|------|------|---------------:|-------------------:|----------|
| easy_1~easy_3（故意伤害/正当防卫/醉驾） | easy | **100% / 100% / 100%** | 100% / 100% / 100% | 简单题两边都搜到 |
| hard_1（故意杀人罪判几年） | hard | **0%** | 0% | 两边都没搜到——"故意杀人罪"关键词在 chunk 里被切散了 |
| hard_2（盗窃罪立案标准） | hard | **100%** | 0% | agentic 自己改了关键词，从"盗窃 立案"扩到了"盗窃 公私财物 数额较大 有期徒刑" |
| hard_3（诈骗罪量刑标准） | hard | **100%** | 0% | 同上——agentic 多搜了两轮才命中 |
| hard_4（醉酒+盗窃前科量刑） | hard | **0%** | 33% | agentic 决定"不搜索就直接答"，non-agentic 反而被动拿到了一些线索 |

**关键观察**：

- **agentic 在 hard_2/hard_3 上大幅胜出**（从 0% 直接到 100%），这就是 ReAct 的核心价值——模型能看到"上一次搜的关键词没命中"，然后**主动改 query**；
- **但 agentic 在 hard_4 上反而掉到 0%**——模型"自信地"决定这道题靠自己已有知识就能答，**一次搜索都没发**。这就是 Agent 的双面性：自主决策既带来优势也带来漏检；
- **non-agentic cite rate 更高**（71% vs 57%）——因为它把检索结果全塞给模型，模型被迫引用；agentic 里模型可能决定"不用引用直接答"；
- 平均延迟 agentic 反而**更快**（18.9s vs 22.3s）——虽然多了几轮 API 调用，但单次搜索更精准，总 token 量更少。

完整轨迹（每轮 ReAct 的 search_laws 调用和返回）见 `results/exp3_8_20261015_143724_812927/trajectories.jsonl`。
