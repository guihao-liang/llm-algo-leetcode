# 09. Attention Mechanism Intro | Attention 机制入门

**难度：** Medium | **标签：** `PyTorch`, `Attention`, `Mask`, `Softmax` | **目标人群：** Chapter 0 入门学习者

## 学习目标

- 手写 scaled dot-product attention 的核心步骤
- 理解 causal mask 的作用
- 为 Chapter 2 的 Attention 题目打底

## 核心练习

- `build_causal_mask`
- `masked_softmax`
- `attention_weights`
- `scaled_dot_product_attention`

## 练习提示

- softmax 前做数值稳定性处理
- mask 应该屏蔽未来 token
- q/k/v 的最后一维通常是特征维
