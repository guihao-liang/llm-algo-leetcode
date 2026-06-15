# 12. Debugging Techniques | 调试技巧

**难度：** Medium | **标签：** `PyTorch`, `Debugging`, `NaN`, `Gradients` | **目标人群：** Chapter 0 入门学习者

## 学习目标

- 学会快速检查张量和梯度是否出现异常
- 理解如何定位 NaN / Inf 问题
- 为后续训练调试和排错打底

## 核心练习

- `tensor_health`
- `has_nonfinite`
- `gradient_l2_norm`
- `find_nonfinite_grad_names`

## 练习提示

- 先看数值是否 finite，再看梯度是否正常
- NaN 通常来自不稳定的 loss、输入或梯度
- 调试时优先缩小问题范围，再扩大复现面
