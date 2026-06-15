# 05. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数

**难度：** Medium | **标签：** `PyTorch`, `Loss`, `Optimizer` | **目标人群：** PyTorch 入门学习者

本页的实战代码可以点击下面的链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/05_PyTorch_Optimizers_and_Loss.md)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

## 学习目标

- 理解 MSE / CrossEntropy 的基本形式
- 掌握 SGD / Adam 的调用方式
- 完成一个最小训练步

## 题目区

1. `mse_loss(pred, target)`
   - 手写最基础的均方误差。
2. `cross_entropy_loss(logits, target)`
   - 手写分类交叉熵的调用和验证。
3. `train_one_step(model, x, target, optimizer)`
   - 把前向、反向、参数更新串成一个最小训练步。

## 练习提示

- 先算 loss，再 backward，再 step
- 使用 `optimizer.zero_grad()` 清空梯度
- 在小样本上验证 loss 是否下降
