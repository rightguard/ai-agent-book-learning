# kv-cache — Prompt Cache + 六种上下文管理模式（实验 2-3）

拆成两个独立研究问题：

- **A. 前缀缓存**：控制实验——长且稳定的材料 + 唯一组标识 + 预热 + 多轮重复测量，看服务端到底缓存了多少 token、TTFT 降了多少；
- **B. 滑动窗口等六种 Agent 上下文管理**：一个"下一步需要哪个文件名只有看上一步工具结果才知道"的调查任务，跑六种上下文管理模式，看滑动窗口会不会真的截断、截断后任务是否还能完成。

## 文件

| 文件 | 说明 |
|------|------|
| `experiment_2_3.py` | 主脚本：A 阶段 controlled cache + B 阶段 ReAct 调查任务，一键跑完 |
| `results/<run_id>/` | `comparison.json`（各模式结果表）+ `cache_ratio.png` / `cache_ttft.png` + `agent_rounds.png`（多轮运行的轮次分布）+ fixture（实验用的模拟工作区文件） |

## 运行

```bash
# 默认 DashScope / qwen3.7-flash
python char2/kv-cache/experiment_2_3.py

# 本地 vLLM
python char2/kv-cache/experiment_2_3.py --provider local

# 只跑缓存阶段
python char2/kv-cache/experiment_2_3.py --phase cache
```

## 六种上下文管理模式

| 模式 | 策略 | 预期效果 |
|------|------|----------|
| `correct` | 完整保留所有历史 | 基线 |
| `dynamic_system` | 动态压缩 system prompt | system prompt 长时收益大 |
| `shuffled_tools` | 工具列表打乱顺序 | 验证工具描述顺序是否影响模型选择 |
| `dynamic_profile` | 动态压缩用户画像/偏好区 | 画像多时减少噪音 |
| `sliding_window` | 只保留最近 N 轮工具结果 | 超预算时截断，但截断后"下一步"信息可能丢失 |
| `text_format` | 把结构化工具结果转成紧凑文本 | 压缩 token，但可能丢失关键结构 |

## 2026-10-13 运行结果摘要

**A. 前缀缓存（以 `correct` 模式为例）**：

| 指标 | 数值 |
|------|------|
| prompt_tokens | 22,702 |
| cached_tokens | **20,480** |
| cache_ratio | **90.2%** |
| TTFT 均值 | 1.99 s（未命中那次 10.8 s 是冷启动） |
| cache_hits / cache_misses | 8 / 0 |

90% 的 prompt token 被服务端前缀缓存命中，TTFT 从冷启动的 10+ 秒降到 0.6~0.8 秒量级。

**B. Agent 调查任务的六模式对照（同一次 run）**：

`comparison.json` 里每种模式都有完整的 `cache_ratio`、`ttft_mean_s`、`task_success`、`agent_iterations`、`repeated_tool_calls`。其中 `correct` 模式 7 轮完成、0 次重复调用；`sliding_window` 模式成功触发了截断（`sliding_window_really_evicted_tool_results: true`），但因为任务链条刚好被截断在一个关键步骤中间，最终任务失败——这就是滑动窗口的典型代价。

## 注意

服务端 `cached_tokens=0` 字段出现不等于缓存没生效；TTFT 下降才是真命中的信号。脚本里已经区分了这两种观测源，报告里的 cache_ratio 优先用 TTFT 计算。
