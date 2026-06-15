# 01. NumPy and Einsum | NumPy 与 Einsum

**难度：** Easy | **标签：** `NumPy`, `Broadcasting`, `Einsum` | **目标人群：** Chapter 0 入门学习者

本页的实战代码可以点击下面的链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/01_NumPy_and_Einsum.md)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

## 学习目标

- 掌握 NumPy 广播与矩阵乘法
- 会用 `einsum` 表达多维张量运算
- 为 Attention 和 Tensor 形状推导打底

## 题目区

1. `build_causal_mask(seq_len)`
   - 生成标准的因果 mask，供后续 Attention 使用。
2. `matmul_einsum(a, b)`
   - 用 `einsum` 重新写出矩阵乘法。
3. `batch_attention_scores(q, k)`
   - 计算 batch 维度下的 Attention 分数。
4. `rms_normalize(x, eps)`
   - 用 NumPy 实现基础的归一化逻辑。

## 练习提示

- 先确认张量 shape，再写公式
- 注意广播维度是否对齐
- `einsum` 适合表达复杂但规律清晰的张量运算
