# context — 上下文消融对照实验（实验 1-1）

同一财务换算任务，在 5 种不同上下文条件下重复跑，对比 Agent 的成功率、工具调用次数和行为模式。用来直观感受"上下文到底帮了 Agent 多少"。

## 任务

> 某公司四个季度营收：Q1 250 万美元、Q2 210 万欧元、Q3 180 万英镑、Q4 3.8 亿日元。调用货币换算工具换成美元，再用计算工具求全年总额和季度平均（保留两位小数）。

脚本内部写死了换算工具 `convert_currency`（固定示例汇率，不是实时汇率）和计算工具 `calculate`（用 AST 安全限制，只允许四则运算）。

## 五组对照

| 模式 | 条件 | 预期行为 |
|------|------|----------|
| `full` | 完整历史、保留推理、保留工具调用声明、保留工具结果 | 正常完成 |
| `no_history` | 每轮只发 system + 原始 user prompt，历史全部丢掉 | 会在第一个子任务上原地打转 |
| `no_reasoning` | 推理内容不回灌到后续请求 | 看看显式推理对长链条决策的影响 |
| `no_tool_calls` | 不给模型声明任何工具 | 模型只能"凭感觉"口算 |
| `no_tool_results` | 调工具了但结果不回灌，模型拿到的是空 observation | 工具白调，模型又开始猜测 |

## 运行

```bash
# 先只检查配置，不消耗 API
python char1/context/context_experiment.py --dry-run

# 跑全部 5 组（默认）
python char1/context/context_experiment.py

# 只挑两组、每组最多 15 轮
python char1/context/context_experiment.py --modes full no_history --max-iterations 15
```

## 结果摘要（2026-10-10 跑出的一轮）

| 实验组 | 状态 | 工具调用 | 重复次数 | 数值正确 | 观测齐全 |
|--------|------|---------:|---------:|:--------:|:--------:|
| full | completed | 5 | 0 | ✅ | ✅ |
| no_history | iteration_limit | 10 | 9 | ❌ | ❌ |
| no_reasoning | completed | 5 | 0 | ✅ | ✅ |
| no_tool_calls | completed | 0 | 0 | ❌ | ❌ |
| no_tool_results | completed | 3 | 0 | ❌ | ❌ |

**关键观察**：

- **no_history** 组把"EUR→USD"那步重复调了 10 次，因为它永远看不到"上次算过 EUR 了"这个事实。这是二次方增长现象的直观复现——历史窗口丢失后，Agent 实际上是在不断重新发现同一个子任务。
- **no_reasoning** 组居然也成功了，但它的后续请求里不再携带 `reasoning_content` 字段。说明"显式把推理链回灌给模型"对这个任务不是硬刚需，但更复杂的规划任务可能就不一样。
- **no_tool_results** 组只调了 3 次工具就停下来了，说明模型"发现自己调了工具但看不到有效结果"就干脆不再调了，而不是硬撑到预算耗尽。

完整终端输出见 [results/result.md](results/result.md)，结构化结果见 `results/context_ablation_*.json`。
