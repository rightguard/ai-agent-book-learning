# 第一章：LLM 推理基础

大型语言模型在推理任何内容之前，会先将文本转换为数字。每一段文本首先被拆分为**词元**，即模型消费的最小单元。随后，每个词元被映射为一个**嵌入向量**，这是一个在高维空间中捕捉其含义的稠密向量。

当用户发送请求时，他们编写的文本被称为**提示词**。模型在该提示词上运行以生成答案的过程称为**推理**。从发送提示词到收到第一个响应之间的时间，就是用户直接感受到的**时延**。

一次最小的推理调用如下所示：

```python
def generate(prompt: str, model) -> str:
    tokens = model.tokenize(prompt)      # split prompt into tokens
    embeddings = model.embed(tokens)     # map each token to an embedding
    output = model.forward(embeddings)   # run inference
    return model.detokenize(output)
```

有两个数字主导着用户体验。第一，提示词中的词元数量，因为更长的提示词会消耗更多的计算资源。第二，第一个词元的时延，因为第一个词元响应缓慢会让整个系统感觉迟钝。在本书中，我们将反复回到这些概念：词元、嵌入向量、提示词、推理和时延。现在就把它们的定义弄清楚，可以避免后续的混淆。
