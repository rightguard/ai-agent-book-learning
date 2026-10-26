# 实验 10-2：Book Translation API 版

- Model: `deepseek-v4-flash-0731`
- Trials: `1`
- Samples: `chapter1.md, chapter2.md, chapter3.md, chapter4.md`
- 设计：A 单 Agent；B Manager+Translation；C 在 B 基础上增加独立审校与有限返工。
- 公平性：每个 trial 预先生成一次术语表，A/B/C 共用相同原文、术语表、提示约束和模型。
- Glossary Agent 的调用成本独立列出，未计入任一 arm；独立部署每条路径时须把共同准备成本加回。
- 最终语义质量采用相同标准的跨组盲评，评估调用不计入 arm 的执行成本；同模型盲评不构成独立质量证据。
- 每个 trial 中 A/B/C 的执行先后顺序轮换，以降低服务负载/缓存造成的次序偏差。
- 这是教学微实验，不是官方第 10-2 章正式复现实验。

## 各次试验

| Trial | Arm | 工程验收 | 调用数 | 输入 Token | 输出 Token | 总耗时(s) | 上下文峰值 Token | 术语遵从代理 | 段落覆盖代理 | 数字遗漏提示 | 返工次数 | 盲评均分 /5 | 盲评报告遗漏 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | A 单 Agent | PASS | 4 | 5472 | 1037 | 13.539 | 2179 | 1.000 | 1.000 | 0 | 0 | 5.000 | 0 |
| 1 | B Manager+Translation | PASS | 5 | 2453 | 1044 | 14.710 | 555 | 0.967 | 1.000 | 4 | 0 | 5.000 | 0 |
| 1 | C Manager+Translation+Proofreading | PASS | 15 | 8868 | 1478 | 29.996 | 1367 | 0.967 | 1.000 | 4 | 0 | 5.000 | 0 |

## 汇总（每 trial 的组内耗时，不含共享 Glossary）

| Arm | 工程验收次数 | 平均调用数 | 平均总 Token | 平均质量分 /5 | 返工总次数 | p50(s) | p95(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A 单 Agent | 1/1 | 4.00 | 6509 | 5.000 | 0 | N/A | N/A |
| B Manager+Translation | 1/1 | 5.00 | 3497 | 5.000 | 0 | N/A | N/A |
| C Manager+Translation+Proofreading | 1/1 | 15.00 | 10346 | 5.000 | 0 | N/A | N/A |

## 共享 Glossary Agent 的成本

- 共用预处理调用数：1
- 共用预处理总 Token：1071
- 共用预处理墙钟时间（所有 trial 合计）：2.672 s
- 各 arm 的总 Token **不含**共用术语提取开销；不得把这种预处理视为免费。

## 统一盲评成本与证据

- 评审模型：`deepseek-v4-flash-0731`
- 评审调用数：12
- 评审阶段累计墙钟时间：19.763 s
- 评审阶段累计 Token：10351
- 盲评只看到当前原文、译文和术语表，不看到实验组名称或内部审校记录。
- 分数为模型主观评分，不是已验证的准确率；遗漏判断必须包含能在原文中核验的原文片段，仍需人工核对。
- 若评审模型与执行模型相同，这是**同源模型盲评**，不能声称为异源独立裁判。

## 分角色调用指标（p50/p95 最少 5 次该角色调用）

| Trial | 阶段/实验组 | 角色 | 调用数 | 输入 Token | 输出 Token | 调用 p50(s) | 调用 p95(s) |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 共享 Glossary | `glossary` | 1 | 962 | 109 | N/A | N/A |
| 1 | A 单 Agent | `single_translation` | 4 | 5472 | 1037 | N/A | N/A |
| 1 | B Manager+Translation | `manager_plan` | 1 | 338 | 24 | N/A | N/A |
| 1 | B Manager+Translation | `translation` | 4 | 2115 | 1020 | N/A | N/A |
| 1 | C Manager+Translation+Proofreading | `global_proofreading` | 2 | 2734 | 133 | N/A | N/A |
| 1 | C Manager+Translation+Proofreading | `manager_decision` | 4 | 755 | 135 | N/A | N/A |
| 1 | C Manager+Translation+Proofreading | `manager_plan` | 1 | 338 | 24 | N/A | N/A |
| 1 | C Manager+Translation+Proofreading | `proofreading` | 4 | 2926 | 159 | N/A | N/A |
| 1 | C Manager+Translation+Proofreading | `translation` | 4 | 2115 | 1027 | N/A | N/A |
| 1 | 统一质量盲评 | `quality_judge` | 12 | 9573 | 778 | 1.601 | 1.704 |

## 验收标准与局限

1. PASS 表示文件/Markdown 结构、必要角色协议、C 组审校问题关闭状态，以及启用时的统一盲评协议均通过；不代表语义翻译质量已通过。
2. `term_doc_hit_rate_proxy` 使用译文字符串匹配，不代表语义层面的术语一致率。
3. `paragraph_coverage_proxy` 和数字遗漏提示不能可靠判断真实的信息遗漏；需要独立评审。
4. 单 Agent 的 prompt_tokens 峰值与 Manager 的峰值使用真实 API usage，模型未返回 usage 时不能用估算值冒充实测。
5. 只在 trial 数不少于 5 时给出 p50/p95；样本过少显示 N/A。
6. B/C 包含 Manager 调度开销；C 还包含审校与返工开销。总成本高不一定意味着架构差，须联合质量判断。
7. A/B/C 在每个 trial 轮换运行顺序；严格研究仍建议扩大样本、随机化与分层比较。
8. api_calls 统计成功返回的模型 completions，attempted_calls 统计脚本尝试；SDK 内部自动重试最多 2 次，未逐个计数。
9. API 没有返回 usage 时，汇总 Token 标为 N/A；observed_* 仅供不完整回执排查，不能当作总消耗。
10. C 组返工预算耗尽而审校仍要求修订时，工程状态是 NOT_PASS。
