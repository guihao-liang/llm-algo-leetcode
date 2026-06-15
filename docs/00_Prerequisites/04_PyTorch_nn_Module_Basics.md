# 04. PyTorch nn Module Basics | PyTorch nn.Module 基础

**难度：** Medium | **标签：** `PyTorch`, `nn.Module`, `Parameters` | **目标人群：** PyTorch 入门学习者

本页的实战代码可以点击下面的链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/04_PyTorch_nn_Module_Basics.md)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

## 题目区

1. `SimpleLinear`
   - 用 `nn.Module` 封装一个最小线性层。
2. `TwoLayerMLP`
   - 组合两个线性层和激活函数，搭一个两层 MLP。
3. `count_parameters(module)`
   - 统计模型中的可训练参数总数。

## 学习目标
- 正确继承 `nn.Module`
- 理解参数注册与 `state_dict`
- 学会组合基础模块构造 MLP

## 练习提示
- 所有可训练参数都应通过 `nn.Parameter` 或子模块注册
- `forward()` 只负责前向传播，不要写训练逻辑
- `state_dict()` 是保存和加载模型的标准接口
