# search-codegen — Responses API + 托管工具（实验 1-3）

两个独立脚本，都是用 DashScope 的 **Responses API** 调用服务端托管工具 `web_search` 和 `code_interpreter`，让搜索和 Python 计算都在模型提供方那边跑。和 `web-search-agent/weather_search.py` 的"本地 function tools"方案形成对照。

## 文件

| 文件 | 说明 |
|------|------|
| `distance.py` | 固定 10 个东盟首都 → 托管搜索拿坐标 → 托管 Python 用 Haversine 算 45 对大圆距离 |
| `bitcoin_analysis.py` | 两轮对话：第一轮强制先澄清模糊 BTC 需求；第二轮补齐后 → 托管搜索 + 托管 Python 算 MA7/MA20/RSI14/MACD 等技术指标 |
| `results/` | 历史运行的 JSON 记录和终端日志 |

## 运行

```bash
# 先预览请求结构，不消耗 API
python char1/search-codegen/distance.py --dry-run
python char1/search-codegen/bitcoin_analysis.py --dry-run

# 真正跑
python char1/search-codegen/distance.py
python char1/search-codegen/bitcoin_analysis.py --interactive  # 第一轮澄清后手动补全需求
```

两个脚本都用 `store=True` + `stream=True`，bitcoin 那个还会把第一轮返回的 `response.id` 塞进第二轮请求的 `previous_response_id`，让模型在服务端直接续上多轮上下文。

## distance.py 结果

- 10 个固定城市，共 $C_{10}^2 = 45$ 对
- 托管 Python 真实算出：**最近一对 = Kuala Lumpur | Singapore，距离 316.03 km**
- Haversine 公式、地球半径 R=6371.0088 km，全部由托管 Python 执行日志打印出来
- 验收标准里还硬编码了"最近对必须是吉隆坡+新加坡、距离与参考值相差 < 5 km"

完整输出见 [results/result_distance.md](results/result_distance.md)。

## bitcoin_analysis.py 结果

两轮 Responses API：

| 阶段 | 行为 |
|------|------|
| 第一轮（澄清） | 模型主动问"数据源选哪个？指标要哪些？"，**没有调用任何工具**（硬约束：含糊需求时必须先澄清） |
| 第二轮（执行） | 用户补齐 → 托管搜索（CoinGecko 等多源）→ 托管 Python 算 MA7/MA20/RSI14/MACD(12,26,9)、区间收益率、最大回撤 → 还画了收盘价走势图 |

2026-10-10 那次运行的最终数值：

| 指标 | Python 算出的值 |
|------|-----------------|
| DATA_POINTS | 76 |
| MA7 | 83,897.71 |
| MA20 | 83,445.07 |
| RSI14 | 43.59 |
| MACD 主线 | 1,036.48 |
| MACD 信号线 | 1,526.92 |
| 近 30 日区间收益率 | +3.77% |
| 期间最大回撤 | -6.04% |

全部由托管 Python 打印 marker 行，脚本再用正则从执行日志里抠出来验收，不是模型口头声称。完整输出见 [results/result_bitcoin.md](results/result_bitcoin.md)。

## 和 web-search-agent 的对比

| 维度 | web-search-agent（本地 tools） | 本目录（托管 tools） |
|------|--------------------------------|----------------------|
| 工具在哪跑 | 本地 Python 函数 | DashScope 服务端容器 |
| Python 计算 | 自己写 AST sandbox | 直接丢给 `code_interpreter` |
| 多轮续接 | 手动拼接 messages 列表 | `previous_response_id` 一把梭 |
| 日志可审计 | 自己存 JSON | 模型方的 streaming event 里带 |
| 可控性 | 完全可控 | 依赖平台能力 |
