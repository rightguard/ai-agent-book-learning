# web-search-agent — 最简 ReAct 天气 Agent

本章最小可运行的 ReAct Agent 示例。工具在本地 Python 实现，通过 Chat Completions 的 `tools=[{...}]` 注册给模型，模型自主决定何时调用、调用哪个参数，然后拿到观察结果后再继续决策。

## 文件

| 文件 | 说明 |
|------|------|
| `weather_search.py` | Agent 主脚本：注册一个 `weather_search` 工具 → 模型 ReAct 循环 → 打印最终回答 |
| `weather_trace.json` | 某次运行的完整轨迹（每轮 tool_call 和 tool 结果） |
| `run_log.md` | 2026-10-10 一次真实运行的终端输出 |

## 运行

```bash
# 必须从仓库根目录启动
python char1/web-search-agent/weather_search.py
```

## 工作方式

```
模型第 1 轮 → 决定"查广州天气" → 本地 Python 函数被调用 → 工具结果返回
模型第 2 轮 → 决定"查深圳天气" → 本地 Python 函数被调用 → 工具结果返回
模型第 3 轮 → 工具够用了 → 生成最终自然语言回答
```

关键设计要点：

- **工具是本地函数**，不是远程托管工具。脚本内部有一个 `weather_search()` 函数调 Open-Meteo 免费天气 API，没有外部 key 需求。
- **ReAct 循环手写**：用 `for step in range(max_iterations)` 驱动，每轮把所有历史消息（包括 tool 结果）都发回给模型，直到模型返回的 `msg.tool_calls` 为空才算"生成完成"。
- **容错与护栏**：搜索次数超过 `config.yaml` 的 `search.max_iterations` 时主动报错；工具异常时把 `{"error": ...}` 作为观察结果回灌，避免 Agent 假装成功。

## 实验结果

2026-10-10 那次运行中，Agent 在 3 轮内先后查询了广州和深圳的三天天气，成功生成 Markdown 表格回答。完整输出见 [run_log.md](run_log.md)，结构化轨迹见 [weather_trace.json](weather_trace.json)。
