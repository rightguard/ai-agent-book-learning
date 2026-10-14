# KV Cache 实验笔记

KV Cache 是保存注意力 Key/Value 张量的机制。
跨请求复用要求已计算的 token 前缀保持稳定。
TTFT 表示首个输出 token 的延迟。
