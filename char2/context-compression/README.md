# context-compression — 六种上下文压缩策略对照（实验 2-10）

同一份固定资料（虚构的"某 AI 公司创始人背景"）、同一个问题、同一个预算（8,500 估计 tokens），用六种不同的上下文压缩策略喂给 LLM，看**客观成功率**和**压缩率**。目的：对照"压缩越多越好"还是"有些压缩策略会把关键信息丢掉"。

## 文件

| 文件 | 说明 |
|------|------|
| `experiment_2_10.py` | 主脚本：加载 DashScope / Moonshot / 本地任一后端，六种策略各跑一次 |
| `results/<run_id>/summary.md` | 自动生成的 Markdown 结果表 |
| `results/<run_id>/comparison.json` | 完整 JSON 记录（每策略的完成状态、客观成功率、最终上下文 token、压缩率、溢出次数） |
| `results/<run_id>/cells/*.json` | 每种策略的原始 LLM 请求/响应/验收 |
| `results/<run_id>/context_token_comparison.png` / `retention_accuracy.png` | 可视化图 |

## 六种压缩策略

| 策略 | 做法 | 预期 |
|------|------|------|
| `no_compression` | 直接把全部原资料塞进 context | 基线，但很可能超预算 |
| `individual` | 每个文档单独让 LLM 先摘要 → 再把各摘要拼起来 | 关键事实可能分散在不同文档，单独摘要后跨文档关联丢失 |
| `combined` | 所有文档一起喂给 LLM，让它一次性输出一份完整摘要 | 压缩率最高，但 LLM 可能"发挥过度" |
| `context_aware` | 带着问题让 LLM 只摘相关部分 | 理论上最好，但模型得先知道"问题是什么" |
| `citations` | 摘要里要求保留引用标记，答案时带引用 | 压缩率中等，可追溯 |
| `windowed` | 把资料按固定窗口切开，滑动喂 LLM 后合并 | 最接近"真实 Agent 每次只看最近一段"的做法 |

## 运行

```bash
python char2/context-compression/experiment_2_10.py

# 本地 vLLM
python char2/context-compression/experiment_2_10.py --provider local
```

## 2026-10-14 运行结果摘要（qwen3.7-flash）

| 压缩策略 | 完成 | 客观成功率 | 压缩率（压缩后/原文） | 最终上下文 tokens | 溢出保护 |
|----------|-----:|-----------:|----------------------:|------------------:|---------:|
| no_compression | ✅ | **0%** | 1.000 | 8,961 | 1 次 |
| individual | ✅ | **100%** | 0.199 | 2,343 | 0 |
| combined | ✅ | **0%** | 0.011 | 330 | 0 |
| context_aware | ✅ | **100%** | 0.082 | 1,349 | 0 |
| citations | ✅ | **0%** | 0.077 | 1,336 | 0 |
| windowed | ✅ | **100%** | 0.534 | 5,758 | 0 |

**关键发现**：

- **combined 压缩最狠但成功率 0%**——LLM 把材料自由"发挥"了，编造了题目里没给的人物关系；
- **context_aware 和 individual 各有 100% 成功率**，说明"让 LLM 带着问题去摘要"或者"让每个片段独立摘要"都比"一次性全开"靠谱；
- **no_compression 居然也 0% 且触发了 1 次预算溢出**——说明模型在超长原资料面前也会被淹没，"多"不等于"好"；
- **citations 带引用标记反而 0%**——可能因为引用标记占用了 token 预算，导致真正的内容被挤掉。

完整分析见 `results/<run_id>/summary.md`。
