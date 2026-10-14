# Chapter 2 — 模型本体、工具调用与护栏验证

本章 7 个实验覆盖 **本地小模型部署、注意力机制、Prefix Cache、Prompt 工程消融、Prompt Injection 攻防、状态栏式 system hint、上下文压缩策略** 七个方向。所有脚本共用仓库根目录的 `config.yaml` + `.env`，**请从仓库根目录启动**。

## 目录一览

| 目录 | 实验 | 主题 |
|------|------|------|
| [local_llm_serving/](local_llm_serving/README.md) | 2-1 | 本地 vLLM 部署 qwen3_0.6b，验证 ReAct 工具调用 + 吞吐 / batching / 前缀缓存三场景 |
| [attention-visualization/](attention-visualization/README.md) | 2-2 | Transformers eager 下抓真实注意力张量，看 0.6B 模型推理 token 和答案 token 的注意力分配 |
| [kv-cache/](kv-cache/README.md) | 2-3 | 前缀缓存命中率实测 + 六种 Agent 上下文管理模式（含滑动窗口截断对任务的影响） |
| [prompt-engineering/](prompt-engineering/README.md) | 2-4 | τ-bench 风格六条 arm 消融：语气 / 噪音 / 工具描述 / 叠加 |
| [prompt-injection/](prompt-injection/README.md) | 2-5 | 三类攻击（direct / indirect / memory）× 四级防御 × 5 次重复，完整攻防矩阵 |
| [system-hint/](system-hint/README.md) | 2-9 | 状态栏式 system hint 消融：时间戳 / 工具计数 / todo / 错误堆栈 / combined，matched task 对照 |
| [context-compression/](context-compression/README.md) | 2-10 | 六种上下文压缩策略（no_compression / individual / combined / context_aware / citations / windowed）客观成功率对照 |

> 注：2-6 / 2-7 / 2-8 在 `char2/` 目录下没有对应子目录，是教程另一章的实验。

## 环境准备

```bash
pip install -r requirements.txt
```

### API Key + config.yaml

根目录 `.env` 至少需要：

```
# DashScope / 通义百炼（实验 2-3 / 2-4 / 2-5 / 2-9 / 2-10）
DASHSCOPE_API_KEY=你的通义百炼 key
QWEN_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 可选：Moonshot Kimi（实验 2-3 / 2-5 / 2-10 支持 --provider moonshot）
MOONSHOT_API_KEY=你的 Kimi key
```

根目录 `config.yaml` 中 `model` 节点通用，各实验如自定义了独立配置（`experiment_2_1` / `experiment_2_2` 等）会优先读取。

### 本地模型（实验 2-1 / 2-2）

这两个实验需要在本机加载一个 **qwen3_0.6b** 小模型。实验 2-1 还要先启 vLLM 服务（见该目录 README），实验 2-2 直接用 Transformers eager 加载。

```bash
# 模型默认放在 /mnt/e/ai-agent-book/qwen3_0.6b
# 必须包含 config.json + tokenizer.json + model.safetensors（或 pytorch_model*.bin）
ls -lah /mnt/e/ai-agent-book/qwen3_0.6b
```

## 快速运行

```bash
# === 不需要本地 vLLM，只要 DashScope Key ===
python char2/kv-cache/experiment_2_3.py
python char2/prompt-engineering/experiment_2_4.py
python char2/prompt-injection/experiment_2_5.py
python char2/system-hint/experiment_2_9.py
python char2/context-compression/experiment_2_10.py

# === 需要先启本地 vLLM（实验 2-1） ===
# 另开一个终端：
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" \
vllm serve /mnt/e/ai-agent-book/qwen3_0.6b \
  --served-model-name qwen3_0.6b --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.50 --max-model-len 6144 \
  --max-num-seqs 8 --max-num-batched-tokens 1024 \
  --enable-chunked-prefill --enforce-eager
# 然后在另一个终端：
python char2/local_llm_serving/experiment_2_1.py

# === 只要本地 HF 模型目录 + GPU（实验 2-2） ===
python char2/attention-visualization/experiment_2_2.py --device cuda
```

## 关键结论索引

每个子目录的 README 里有完整结果表和分析，这里按实验编号把**本次（2026-10-13 / 14）运行得到的核心结论**集中列出来，方便扫一眼就知道本章都证明了什么：

| 实验 | 问题 | 核心结论 |
|------|------|----------|
| 2-1 | 本地 0.6B 小模型能不能跑 Agent | 能完整跑通 ReAct + 工具调用；前缀缓存命中率 ~70% 后 TTFT 从冷启动 10s 降到 0.6-0.8s；生成吞吐 15-20 tokens/s |
| 2-2 | 模型注意力到底堆在哪些 token 上 | layer_27 顶层 **83.2% 注意力粘在序列开头 Sink token**，只有不到 17% 跑到实际内容上；这是推理效率低的直观证据 |
| 2-3 | Prompt Cache 真实命中率 + 滑动窗口截断 | 前缀缓存命中 **90.2%** prompt tokens；滑动窗口确实截断，但被截断的 Agent 任务（下一步依赖被截断掉的工具结果）**失败** |
| 2-4 | 提示词工程到底有没有用 | 语气改写（Trump / casual）和 wiki 噪音对**成功率零影响**；删掉工具 description 后步骤从 10→12、工具调用从 5→7，但还是能歪打正着地完成 |
| 2-5 | Prompt Injection 的真正威胁在哪 | direct / indirect 两类**全 0 攻**；memory 类在无防御下 **100% 中招**（模型把共享笔记当成了团队规范），加 source tag 降到 20% |
| 2-9 | system prompt 末尾加一段"状态栏"有啥用 | matched task 机制下每种 hint 都跑 5 个 case × 2（实验组 / disabled）；状态栏在**时间排序、按时间资源分类、工具调用预算控制**这三类任务上有显著收益 |
| 2-10 | 上下文压缩策略哪家强 | `context_aware` 和 `individual` 各 100% 成功率；`combined` 压缩率最狠（1.1%）但成功率 **0%**；超长原资料（不压缩）也 0% 且触发预算溢出 |

## 小提示

- **所有实验都是安全隔离的**：`prompt-injection` 的 send_email 永远只写 `outbox.jsonl`；`local_llm_serving` 的工具是本地 Python 函数，不访问真 Open-Meteo；`prompt-engineering` 用的是 pin 住 commit 的迷你航班环境。
- **真实 Prompt Cache 命中率请同时看 TTFT 下降和服务端 `cached_tokens` 字段**；两者对不上时以 TTFT 为准（实验 2-3 已这样处理）。
- **部分结论只在"单次任务 + 单模型"下成立**（如 2-4 的六条 arm 全 100%）；要得出可推广结论需要多任务多模型 campaigns。
- char1 根 README 里的"两种 API 模式"对照表对本章依然成立：2-1（本地 vLLM）+ 2-2（Transformers eager）属于本地推理；2-3 / 2-4 / 2-5 / 2-9 / 2-10 全部走 DashScope Chat Completions。
