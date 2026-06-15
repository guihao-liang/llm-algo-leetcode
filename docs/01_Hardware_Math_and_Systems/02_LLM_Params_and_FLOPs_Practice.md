# 02. LLM Params and FLOPs Practice | LLM Params and FLOPs - 计算练习

**难度：** Medium | **标签：** `参数量`, `FLOPs`, `训练时间` | **目标人群：** 算法 / Infra 学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本练习配套理论文档：[02_LLM_Params_and_FLOPs.md](./02_LLM_Params_and_FLOPs.md)

建议先阅读理论文档，再来完成本练习。理论页负责解释参数量和 FLOPs 怎么来，Notebook 负责把训练时间和场景判断真正算出来。

---

## 🎯 学习目标

- 拆解 Transformer 的参数组成
- 计算训练与推理 FLOPs
- 估算给定硬件配置下的训练时间
- 把理论公式转成可复用的计算函数

## 本节如何和 Notebook 配合

这一节建议先和 `[02_LLM_Params_and_FLOPs.md](./02_LLM_Params_and_FLOPs.md)` 一起学。
先完成题目区中的推导和计算，再看参考答案区。Notebook 的目标是把“参数量 -> FLOPs -> 时间”这条链条串起来。



## 题目区

### TODO 概览

- Part 1: Transformer 参数量计算

- Part 2: 训练与推理 FLOPs 计算

- Part 3: 场景应用



## Part 1: Transformer 参数量计算

实现一个函数，按照 `Embedding + Attention + SwiGLU FFN + LayerNorm + LM Head` 的方式估算 Transformer 参数量。

```python
def calculate_transformer_params(vocab_size, hidden_dim, num_layers, intermediate_size=None, tie_embeddings=False):
    if intermediate_size is None:
        intermediate_size = 4 * hidden_dim

    embedding_params = vocab_size * hidden_dim
    attention_params = num_layers * (4 * hidden_dim * hidden_dim)
    ffn_params = num_layers * (3 * hidden_dim * intermediate_size)
    layernorm_params = num_layers * (2 * hidden_dim)
    lm_head_params = 0 if tie_embeddings else vocab_size * hidden_dim

    total_params = embedding_params + attention_params + ffn_params + layernorm_params + lm_head_params
    return total_params
```


```python
def test_calculate_transformer_params():
    try:
        result = calculate_transformer_params(1000, 64, 2, 256, tie_embeddings=True)
        assert result == 195328, f"错误：期望 195328，实际 {result}"

        result = calculate_transformer_params(1000, 64, 2, 256, tie_embeddings=False)
        assert result == 259328, f"错误：期望 259328，实际 {result}"

        result = calculate_transformer_params(32000, 4096, 32, 11008, tie_embeddings=False)
        assert result > 6000000000, f"错误：LLaMA 级别模型参数量应大于 6B，实际 {result}"

        print("✅ 参数量函数测试通过！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_transformer_params()
```

### 练习 1.2: 估算 LLaMA-7B 的参数量

使用上面的函数估算一个 7B 级模型的大致参数规模，并观察各模块的占比。

```python
vocab_size = 32000
hidden_dim = 4096
num_layers = 32
intermediate_size = 11008

total_params = calculate_transformer_params(vocab_size, hidden_dim, num_layers, intermediate_size, tie_embeddings=False)
print(f"估算参数量: {total_params / 1e9:.2f}B")
print("提示：不同实现会因为是否共享词嵌入、是否计入偏置而略有差异。")
```

---
## Part 2: 训练与推理 FLOPs 计算

大模型训练常用近似公式：`训练 FLOPs ≈ 6 × 参数量 × token 数`。

```python
def calculate_training_flops(num_params_b, num_tokens, flops_per_param_token=6):
    return num_params_b * 1_000_000_000 * num_tokens * flops_per_param_token
```


```python
def test_calculate_training_flops():
    try:
        result = calculate_training_flops(7, 1_000_000_000_000)
        assert result == 42_000_000_000_000_000_000_000, f"错误：期望 4.2e22，实际 {result}"

        result = calculate_training_flops(1, 1_000_000, flops_per_param_token=6)
        assert result == 6_000_000_000_000_000, f"错误：期望 6e15，实际 {result}"

        print("✅ FLOPs 函数测试通过！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_training_flops()
```

### 练习 2.2: 估算训练时间

给定 GPU 的理论算力、GPU 数量和利用率，估算完成训练需要的时间。

```python
def estimate_training_time(num_params_b, num_tokens, gpu_tflops, num_gpus, efficiency=0.35):
    total_flops = calculate_training_flops(num_params_b, num_tokens)
    effective_flops = gpu_tflops * 1e12 * num_gpus * efficiency
    hours = total_flops / effective_flops / 3600
    return hours
```


```python
def test_estimate_training_time():
    try:
        result = estimate_training_time(1, 1_000_000, 1, 1, 1.0)
        assert abs(result - 1.6666666666666667) < 1e-9, f"错误：期望约 1.6667 小时，实际 {result}"

        result = estimate_training_time(7, 1_000_000_000_000, 312, 8, 0.35)
        assert result > 1000, f"错误：大模型训练时间应远大于 1000 小时，实际 {result}"

        print("✅ 训练时间函数测试通过！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_estimate_training_time()
```

---
## Part 3: 场景应用

把参数量、FLOPs 和训练时间串联起来，感受公式在工程中的作用。

```python
scenarios = [
    ('1x A100 80GB', 312, 1, 0.35),
    ('8x A100 80GB', 312, 8, 0.35),
    ('8x H100 80GB', 500, 8, 0.45),
]

print('LLaMA-7B 训练时间粗估（1T tokens）：')
print('-' * 70)
for name, gpu_tflops, num_gpus, eff in scenarios:
    hours = estimate_training_time(7, 1_000_000_000_000, gpu_tflops, num_gpus, eff)
    print(f"{name:<16} {hours:>12.1f} 小时  ({hours/24:>8.1f} 天)")
```

---

## 🎓 总结

通过本练习，你应该能把 `参数量 → FLOPs → 训练时间` 这一条链路串起来。

**下一步：** 学习 [03. GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.md)，进一步理解显存层次和带宽瓶颈。
---

🛑 **STOP HERE** 🛑



## 参考答案与解析

下面给出参考实现与解析。建议先独立完成题目区，再对照参考答案。
