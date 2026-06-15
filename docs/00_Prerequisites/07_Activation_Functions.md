# 07. Activation Functions | 激活函数

**难度：** Easy | **标签：** `PyTorch`, `Activation`, `GELU`, `SiLU` | **目标人群：** Chapter 0 入门学习者

## 学习目标

- 理解 ReLU、GELU 和 SiLU 的常见写法
- 学会把激活函数抽象成可复用的小工具
- 为后续 Transformer/MLP 代码阅读打底

## 核心练习

- `relu`
- `gelu_exact`
- `silu`
- `activation_summary`

## 练习提示

- 先确认公式，再写实现
- 尽量对齐 PyTorch 官方实现做单测
- 激活函数属于逐元素运算，注意输入输出 shape 保持一致
