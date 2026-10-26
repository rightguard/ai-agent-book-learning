# 第2章：Transformer 与注意力

现代语言模型建立在 **Transformer** 架构之上。其核心思想是 **注意力**：模型并非严格从左到右读取序列，而是让每个词元都能查看其他所有词元，并决定哪些词元更为重要。这就是为什么 Transformer 能够将代词与许多词元之前出现的名词关联起来。

注意力机制作用于每个词元的 **嵌入向量**。对于每个词元，模型会计算三个向量——查询、键和值——并用它们来权衡每个词元应多大程度地关注其他词元。

```python
def attention(query, key, value):
    scores = query @ key.T          # similarity between tokens
    weights = softmax(scores)       # attention weights
    return weights @ value          # weighted embedding
```

由于注意力机制会将每个词元与其他所有词元进行比较，其计算成本会随着提示词变长而迅速增加。这是我们在下一章将要解决的时延问题的根本原因。尽管如此，注意力正是 Transformer 强大能力的来源：在推理过程中，它使模型能够在整个提示词范围内灵活地路由信息，而非通过固定的流水线。
