# 第 1 章：LLM 推理基础

大语言模型在推理任何内容之前，会先将文本转换为数字。每段文本首先被拆分为**词元**，即模型消费的最小单元。随后，每个词元被映射为**嵌入向量**，一个在高维空间中捕捉其含义的稠密向量。

当用户发送请求时，他们输入的文本被称为**提示词**。模型在该提示词上运行以生成答案的过程称为**推理**。从发送提示词到收到第一个响应之间的时间，就是用户直接感受到的**时延**。

一次最小的推理调用如下所示：

```python
def generate(prompt: str, model) -> str:
    tokens = model.tokenize(prompt)      # split prompt into tokens
    embeddings = model.embed(tokens)     # map each token to an embedding
    output = model.forward(embeddings)   # run inference
    return model.detokenize(output)
```

有两个数字主导着用户体验。首先是提示词中的词元数量，因为更长的提示词需要更多的计算量。其次是第一个词元的时延，因为第一个词元响应缓慢会让整个系统感觉迟钝。在本书中，我们会反复回到这些概念：词元、嵌入向量、提示词、推理和时延。现在就把它们的定义弄清楚，以后就不会混淆了。
