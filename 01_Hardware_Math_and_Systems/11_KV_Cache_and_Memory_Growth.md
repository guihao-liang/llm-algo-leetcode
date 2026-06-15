# 11. KV Cache 与显存增长 | KV Cache and Memory Growth

**难度：** Medium | **标签：** `推理显存`, `Attention`, `KV Cache` | **目标人群：** 准备进入 Chapter 2 / 3 的学习者

这一页把 Chapter 1 的显存估算，接到 Chapter 2 的长上下文推理问题上。重点只有一个：KV cache 会随着上下文变长而持续增长。

## 为什么这一节是前置页

- Chapter 1 已经讲过参数量、显存和 Attention 代价
- Chapter 2 会直接遇到 MHA/GQA、FlashAttention 和 PagedAttention
- 这一页先把“推理时显存为什么涨”说清楚

## 你应该先建立的直觉

### 1. 训练显存和推理显存不是一回事

训练时，显存大头通常是：
- 参数
- 梯度
- 优化器状态
- 激活值

推理时，显存大头通常变成：
- 参数
- KV cache
- 临时激活和 workspace

也就是说，推理时最容易持续增长的部分，不是梯度，而是 KV cache。

### 2. KV cache 会随着上下文长度增长

在自回归生成里，每生成一个 token，都要把这一轮的 Key 和 Value 保存下来，供后续 token 复用。

这带来一个直接结果：
- 上下文越长，cache 越大
- batch 越大，cache 越大
- 层数越多，cache 越大
- head 数和 head dim 越大，cache 也越大

### 3. 长上下文推理不是“单次算得慢”，而是“历史状态越来越贵”

这也是很多优化的出发点：
- FlashAttention 主要解决计算和中间矩阵存储
- PagedAttention 主要解决 KV cache 的组织和分配
- Quantization 主要解决参数和部分缓存的内存压力

## 一个最常见的估算方式

对 decoder-only 模型来说，KV cache 可以粗略理解为：

```text
KV Cache 规模 ∝ 层数 × 序列长度 × batch size × hidden/head 相关维度
```

你不用在这一页死记完整公式，关键是知道：
- 序列长度翻倍，cache 就明显变大
- batch 翻倍，cache 也会明显变大
- 模型越深，cache 也越难忽略

## 常见误区

- 把 KV cache 当成训练时的梯度缓存
- 只看参数量，不看推理时的历史状态
- 以为长上下文只是“attention 计算更慢”，其实首先是显存压力更大
- 把“能生成”误当成“能稳定长时间生成”

## 这一页学完后，你应该能回答

- 为什么推理时显存会随着生成长度增长
- 为什么长上下文推理会越来越难做
- 为什么 vLLM 会强调 PagedAttention
- 为什么 Chapter 2 的 `Attention`、`FlashAttention` 和 `PagedAttention` 要连着学

## 和后续章节的联系

- **Chapter 2: Attention MHA/GQA**  
  这里会直接看到注意力头数、KV 共享和缓存大小的关系

- **Chapter 2: FlashAttention**  
  你会看到计算优化如何降低中间矩阵的压力，但它不等于自动解决 KV cache 组织问题

- **Chapter 2: vLLM PagedAttention**  
  这里会把 KV cache 的管理变成真正的工程问题

## 小结

这一页的作用很简单：
- 先让你知道 KV cache 是什么
- 再让你知道它为什么会长
- 最后让你知道为什么 Chapter 2 要专门处理它

能把“参数显存”和“推理缓存”区分开，就说明这一页的目标基本达成，后面可以继续看 Chapter 2。
