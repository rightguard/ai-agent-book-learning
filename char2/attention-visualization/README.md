# attention-visualization — 本地注意力热力图（实验 2-2）

直接拿本地 Qwen3-0.6B 模型在 **Transformers eager** 模式下跑一次真实推理，把所有层、所有 head 的注意力矩阵导出来，画成热力图。重点看"推理区 vs 答案区"的注意力分配——为什么小模型在推理 token 上花了那么多算力，但最后答案只有十几 token。

## 文件

| 文件 | 说明 |
|------|------|
| `experiment_2_2.py` | 主脚本：加载本地模型 → 生成推理序列 → 抓每层注意力张量 → 保存热力图 PNG 和原始矩阵 |
| `results/<run_id>/` | `attention_matrices.npz`（所有层所有 head 的注意力张量，可离线分析）+ `beijing_attention.png`（简单提示词）+ `reasoning_attention.png`（推理提示词）+ `manifest.json` |

## 运行

```bash
python char2/attention-visualization/experiment_2_2.py --device cuda
# 或者 CPU（慢很多）
python char2/attention-visualization/experiment_2_2.py --device cpu
```

脚本内部写死了两类 prompt：

| prompt | 目的 |
|--------|------|
| 简单："北京 的 天气 怎么样" | 无推理需求，观察正常问答下注意力分布 |
| 推理："请先认真推理，再用一句简洁的话回答：17 × 23 等于多少？" | 故意让模型先吐一堆 `<think>` 推理 token，再出短答案 |

默认只看 3 层：`layer_0`（底层）、`layer_13`（中层）、`layer_27`（最后一层），每个层取最后一个 head（`head=-1`）。

## 2026-10-13 运行结果摘要

推理 prompt 生成了 **368 token 的思考区 + 14 token 的答案区**。

| 层 | Sink token 平均占比 | 备注 |
|----|---------------------|------|
| layer_0 | 14.7% | 底层相对均匀 |
| layer_13 | 73.7% | 中层开始高度关注 Sink |
| layer_27 | **83.2%** | 顶层绝大多数注意力都堆在开头几个 token |

Sink token（就是 `<think>`、`你`、`好` 这些序列开头的 token）越往高层越"粘"——83% 的注意力没跑到 300 多个推理 token 上，而是在开头打转。这是小模型推理效率低的直观证据：它们学会了在 Sink 位置做某种"全局汇聚"，但牺牲了对长序列细节的关注。

## 产出文件

- `attention_matrices.npz`：`dict[layer_idx → np.ndarray[heads, seq, seq]]`，可直接 `np.load()` 离线分析
- `beijing_attention.png` / `reasoning_attention.png`：matplotlib 画出来的热力图（x/y 轴是 token，颜色深浅是注意力权重）
- `manifest.json`：模型权重 SHA256、配置快照、prompt、seed 等溯源信息
