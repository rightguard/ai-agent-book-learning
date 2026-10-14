# system-hint — 状态栏式 system hint 消融（实验 2-9）

同一个受限文件操作 Agent，**matched task 机制**下——每组 hint 都配一份**完全相同**的任务（文件排序、按时间复制资源、分类文档等），保证对照组 `disabled` 和实验组只差一个 hint 开关。目的：看"在 system prompt 末尾附加一段运行时状态栏"这件事，对 Agent 完成质量有多大提升。

## 文件

| 文件 | 说明 |
|------|------|
| `experiment_2_9.py` | 主脚本：DashScope 后端 + 6 种 hint 机制，一键跑完 matched tasks |
| `results/<run_id>/cases/*.json` | 每个 case × hint 组合的完整执行记录（含工具调用轨迹和验收） |
| `results/<run_id>/sandboxes/` | 每个 case 的隔离工作区（documents/、resources/、records/）+ `initial_state.json` |
| `results/<run_id>/protocol.json` | 完整的 conditions 定义（每种 hint 对应哪些标签）和 case 清单 |

## 六种 system hint 机制

每个 hint 其实就是一段动态拼到 system prompt 末尾的状态栏：

| hint | 状态栏里有什么 | 对应问题 |
|------|----------------|----------|
| `disabled` | 什么都不加 | **对照组** |
| `timestamps_raw` | 当前时间戳 | "模型忘了现在几点了？" |
| `timestamps_guided` | 时间戳 + "请先查最新修改时间再排序" | "模型忘了**怎么**用时间戳？" |
| `tool_counter` | "你这轮已经调了 3 次工具，当前预算 10" | "模型陷入反复重试？" |
| `todo_list` | 已完成步骤列表 + 下一步 | "模型忘了自己做到哪了？" |
| `detailed_errors` | 上一轮失败工具的详细错误堆栈 | "模型只看到 'error=true' 不知道为啥错？" |
| `combined` | 上述所有 hint 全开 | 上限探索 |

每个 hint 都有 5 个 case（ts-01~05 / ctr-01~05 / todo-01~05 等），总共约 **5×5×2=50 个执行单元**。

## 运行

```bash
python char2/system-hint/experiment_2_9.py
```

脚本会把每个 case 的执行记录分别存成 `cases/<hint>__<case_id>__<hint>.json` 和 `cases/<hint>__<case_id>__disabled.json`（因为是 matched 机制，同一个 case 会被 disabled 和实验组各跑一次）。每个沙箱目录里还有 `initial_state.json` 记录了任务开始时文件/资源的原始状态。

## 关键设计

- **matched task 机制**：disabled 和实验组跑完全相同的沙箱、完全相同的任务指令，唯一差异就是 system prompt 末尾那段 hint 状态栏。这样才能把因果关系对齐——成功率变化几乎只来自 hint。
- **独立沙箱**：50 个执行单元各有独立 `sandboxes/<hint>__<case_id>/`，不会互相污染文件状态。
- **limited tool 集**：Agent 只能用 `find`、`read_file`、`grep`、`write_file`、`copy_file` 这几个受限工具，不跑 pip install，不访问外网。
