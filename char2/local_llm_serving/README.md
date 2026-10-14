# local_llm_serving — 本地 vLLM 部署 + 工具调用（实验 2-1）

用 **vLLM 在本机（WSL）服务一个 0.6B 的 Qwen3 小模型**，让它走完整 ReAct 工具调用循环（time + weather 两个函数），同时测吞吐、批处理、前缀缓存三个维度。目的：验证本地小模型能不能撑住 Agent 场景，以及 vLLM 在并发/缓存下的表现。

## 文件

| 文件 | 说明 |
|------|------|
| `experiment_2_1.py` | 主脚本：工具调用 + 吞吐 + batching + cache 四个场景，一键跑完 |
| `result.md` | 2026-10-13 一次真实运行的终端输出（含 vLLM serve 日志和实验结果） |
| `results/<run_id>/` | 结构化 JSON 和 PNG（吞吐/首 token 延迟/缓存命中率曲线） |

## 先启 vLLM 服务（WSL）

```bash
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" \
vllm serve /mnt/e/ai-agent-book/qwen3_0.6b \
  --served-model-name qwen3_0.6b \
  --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.50 \
  --max-model-len 6144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --enable-chunked-prefill \
  --enforce-eager
```

关键参数：
- `--tool-call-parser hermes` 让 vLLM 正确解析 Hermes 风格的 `<tool_call>{...}</tool_call>` 工具调用标签；
- `--enable-chunked-prefill` 让长 prompt 分批进显存；
- `--enforce-eager` 关闭 torch.compile / CUDAGraph，换取可重复的性能数字（不追求极速）。

## 跑实验

```bash
python char2/local_llm_serving/experiment_2_1.py
```

脚本会依次跑四个场景：

| 场景 | 做什么 |
|------|--------|
| **tools** | ReAct 工具调用循环：并行调 `get_current_time` + `get_current_temperature`，本地 Python 里实现两个工具 |
| **raw** | 直接打印模型流出来的原始 token（含 `<think>`、`<tool_call>` 标签），对照"喂给 vLLM 后解析"和"原始字符串"之间的差别 |
| **throughput** | 固定 prompt 重复 5 次，测平均生成吞吐（tokens/s）和首 token 延迟（TTFT） |
| **batching** | 并发度 1 / 2 / 4 / 8 各跑若干次，看 batching 收益曲线 |
| **cache** | 相同前缀请求重复，测 vLLM 前缀缓存（prefix caching）命中率 |

## 2026-10-13 运行结果摘要

- **ReAct 成功**：两轮完成，工具调用并行发起、并行返回，模型最终正确报告了温哥华时间 + 10.2°C 气温
- **GPU KV cache 1.8 GiB** → 约 16,864 token 容量；单次请求 6,144 token → 理论并发上限 2.74x
- **前缀缓存命中率从 0% 爬升到 ~70%**（第 4~5 次请求开始稳定命中）
- **生成吞吐**：单请求 ~15-20 tokens/s；4 并发时瞬时峰值 56 tokens/s
- **0.6B 小模型完整跑通工具调用链**，说明本地部署 Agent 做教学演示是完全可行的

完整日志见 [result.md](result.md)。
