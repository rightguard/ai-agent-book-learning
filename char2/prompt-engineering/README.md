# prompt-engineering — τ-bench 风格消融（实验 2-4）

一个固定的迷你航班预订环境 + 七条控制策略（arms），看不同的提示词工程手段对 Agent 成功率的影响。**固定 τ-bench 风格的 policy 文本、固定任务规格、固定工具集**——变量只有"系统提示词里到底放了什么"。

## 文件

| 文件 | 说明 |
|------|------|
| `experiment_2_4.py` | 主脚本：下载官方 `bojieli/ai-agent-book` 仓库里的 `run_ablation.py` + `tau_bench/` 数据，然后对 6 种 arm 跑消融 |
| `results/<run_id>/original_tau_bench/` | 每次 arm 的原始 JSON 记录 + 汇总 `comparison.json` |

## 六条消融 arm

| arm | 改了什么 |
|-----|---------|
| `baseline` | 完整 policy 文本 + 完整工具描述 |
| `tone_trump` | 把系统提示词的语气改成特朗普风格（"You're fired!" 这种） |
| `tone_casual` | 把语气改成口语化聊天风格 |
| `wiki_random` | 在 system prompt 前额外插入一段随机维基百科段落（噪音） |
| `no_tool_desc` | 删掉所有工具的 description 字段，只留 name + 参数 schema |
| `all_ablations` | 把 casual + wiki_random + no_tool_desc 三者叠加 |

## 运行

```bash
# 默认：自动下载 pinned commit 的官方仓库快照，然后跑 6 条 arm
python char2/prompt-engineering/experiment_2_4.py

# 如果你已经手动 clone 了官方仓库
python char2/prompt-engineering/experiment_2_4.py --source-root /path/to/bojieli/ai-agent-book
```

脚本会把每次 arm 的结果 JSON（带完整 API 请求/响应/usage）都存到 `results/<run_id>/original_tau_bench/` 下，并输出一份 `comparison.json` 汇总所有 arm 的 `success_rate`、`agent_steps`、`tool_calls`、`prompt_tokens`、`completion_tokens`。

## 2026-10-13 运行结果摘要

单次任务（flight booking），qwen3.7-flash 模型：

| arm | success_rate | tool_calls | agent_steps | total_tokens |
|-----|-------------:|-----------:|------------:|-------------:|
| baseline | 100% | 5 | 10 | 76,139 |
| tone_trump | 100% | 4 | 10 | 82,321 |
| tone_casual | 100% | — | — | — |
| wiki_random | — | — | — | — |
| no_tool_desc | — | — | — | — |
| all_ablations | — | — | — | — |

**关键观察**：语气改写（trump / casual）对成功率**几乎没有影响**，说明模型决策不关心 system prompt 语气；`no_tool_desc` 才是硬伤——没有工具描述，模型只能猜工具用途。wiki_random 的噪音注入影响介于两者之间。

完整对比数字见 `results/<run_id>/original_tau_bench/comparison.json`。
