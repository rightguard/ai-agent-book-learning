# learning-from-experience — Q-learning vs LLM Agent 寻宝对照（实验 8-2）

同一个五房间文字寻宝游戏，让**离线训练的 Q-learning** 和**零样本 LLM Agent** 分别跑一局，对比强化学习和大模型 in-context learning 两种"决策能力来源"。

## 游戏规则

五个房间：entrance → hallway → guard_room → treasure_room，外加 storage 储物间。通关路径：

1. 在 entrance 拿到 rusty sword（铁剑）
2. 走 east 进 storage，拿 red key 和 magic crystal
3. 回到 hallway，用 red key 打开北边锁门
4. 进 guard_room，尝试用铁剑打怪失败 → 用道具 craft 合成 silver sword
5. 用银剑击败守卫 → 进 treasure_room → 拿 dragon's treasure → **胜利**

最多 50 步，每步根据动作结果给正负 reward。整个游戏环境**内嵌在脚本里**，无外部依赖。

## 文件

| 文件 | 说明 |
|------|------|
| `qlearning_experiment.py` | 单独跑 Q-learning：训练 10000 局 + 评估 100 局，输出 Q 表、learning curve |
| `llm_comparison.py` | **一键对照**：先跑 Q-learning 基线（训练 + 评估），再让 Qwen 零样本跑 1 局，自动打印对照结果并保存 manifest |
| `results/<run_id>/` | 每次运行的完整记录：`comparison.json`、`learning_curve.csv`、`learning_curve.png`、`manifest.json`、`rl_agent.pkl` 等 |

## 运行

```bash
# 一键对照（推荐）
python char1/learning-from-experience/llm_comparison.py

# 只跑 Q-learning
python char1/learning-from-experience/qlearning_experiment.py
```

## 2026-10-11 运行结果

| 指标 | Q-learning 基线 | Qwen 零样本 |
|------|----------------|-------------|
| 训练 / 尝试次数 | 10000 局训练 + 100 局评估 | 1 局 |
| 胜率 | **100.0%** | 1 局里**成功** |
| 平均步数 | 12.0 步 | 24 步 |
| 用时 | 训练 9.96 秒 + 评估 < 1 秒 | 30.8 秒 |
| 调用成本 | 无（纯本地） | 24 次 API 调用，18536 tokens |

Qwen 那局的实际轨迹：24 步通关，前几步探索有点绕（先试着用铁剑打守卫失败 → 再用道具 craft 合成银剑 → 再打守卫），但和人类第一次玩的行为模式几乎一致。完整终端日志见 [results/results.md](results/results.md)，结构化对照见 `results/<run_id>/comparison.json`。

## 核心问题

- Q-learning 在 10000 局训练后 Q 表收敛到 142 个 state，评估集 100 局全胜；但它完全不懂"为什么"，只是把 state-action → value 背下来了。换个新房间布局就要重训。
- Qwen 没有任何显式训练，靠预训练里见过的"房间-物品-动作"世界知识直接做决策；但它**不稳定**——第一次跑可能就失败，也可能像这次一样刚好成功，没法像 RL 那样给一个稳定的胜率数字。

这个对照正好对应 Agent 工程里两条路线的取舍：要不要花算力给每个环境单独训一个专用策略，还是用通用大模型直接推理。
