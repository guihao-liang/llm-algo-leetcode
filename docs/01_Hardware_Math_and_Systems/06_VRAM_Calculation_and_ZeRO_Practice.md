# 06. VRAM Calculation and ZeRO Practice | VRAM Calculation and ZeRO - 计算练习

**难度：** Hard | **标签：** `显存计算`, `ZeRO`, `梯度累积` | **目标人群：** 训练 / 分布式学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本练习配套理论文档：[06_VRAM_Calculation_and_ZeRO.md](./06_VRAM_Calculation_and_ZeRO.md)

建议先阅读理论文档，再来完成本练习。理论页负责讲清 16Φ、ZeRO 和激活值显存，Notebook 负责把这些公式真正算出来。

---

## 🎯 学习目标

- 计算 DDP 训练的显存占用
- 计算 ZeRO-1 / ZeRO-2 / ZeRO-3 的显存节省
- 反推给定显存和卡数下可训练的最大模型规模
- 把显存公式转成可复用的函数

## 本节如何和 Notebook 配合

这一节建议和理论文档一起学：

- 先看理论页，理解 16Φ、ZeRO 和激活值显存的推导
- 再做 Notebook，把 DDP、ZeRO 和最大模型规模真正算一遍
- Notebook 里的测试用来确认你不是“看懂了”，而是真的“会估算、会反推”

### 练习目标

- 计算 DDP、ZeRO-1、ZeRO-2、ZeRO-3 的显存占用
- 反推给定显存条件下可训练的最大模型规模
- 用测试确认公式结果与理论页的结论一致

## Part 1: DDP 显存计算

在 DDP 下，每张卡都保存完整模型、完整梯度和完整优化器状态。对于 Adam + FP16/BF16 训练，经验公式是 **16Φ**。

```python
def calculate_ddp_memory(num_params_b, model_dtype='fp16', optimizer='adam'):
    model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
    gradient_bytes = model_bytes
    optimizer_bytes = {'adam': 12, 'sgd': 4}[optimizer]
    total_bytes = model_bytes + gradient_bytes + optimizer_bytes
    return num_params_b * total_bytes
```


```python
def test_calculate_ddp_memory():
    try:
        result = calculate_ddp_memory(7, 'fp16', 'adam')
        assert result == 112, f"错误：期望 112 GB，实际 {result} GB"

        result = calculate_ddp_memory(7, 'fp16', 'sgd')
        assert result == 56, f"错误：期望 56 GB，实际 {result} GB"

        print("✅ DDP 显存函数测试通过！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_ddp_memory()
```

## Part 2: ZeRO 显存计算

- ZeRO-1：切分优化器状态
- ZeRO-2：切分优化器状态和梯度
- ZeRO-3：切分参数、梯度和优化器状态

```python
def calculate_zero_memory(num_params_b, zero_stage, num_gpus, model_dtype='fp16', optimizer='adam'):
    model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
    gradient_bytes = model_bytes
    optimizer_bytes = {'adam': 12, 'sgd': 4}[optimizer]

    if zero_stage == 0 or zero_stage == 'ddp':
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes
    elif zero_stage == 1:
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes / num_gpus
    elif zero_stage == 2:
        bytes_per_param = model_bytes + gradient_bytes / num_gpus + optimizer_bytes / num_gpus
    elif zero_stage == 3:
        bytes_per_param = (model_bytes + gradient_bytes + optimizer_bytes) / num_gpus
    else:
        raise ValueError('zero_stage must be 0/ddp, 1, 2 or 3')

    return num_params_b * bytes_per_param
```


```python
def test_calculate_zero_memory():
    try:
        result = calculate_zero_memory(7, 1, 8, 'fp16', 'adam')
        assert abs(result - 38.5) < 1e-9, f"错误：ZeRO-1 应该是 38.5 GB，实际 {result} GB"

        result = calculate_zero_memory(7, 2, 8, 'fp16', 'adam')
        assert abs(result - 26.25) < 1e-9, f"错误：ZeRO-2 应该是 26.25 GB，实际 {result} GB"

        result = calculate_zero_memory(7, 3, 8, 'fp16', 'adam')
        assert abs(result - 14) < 1e-9, f"错误：ZeRO-3 应该是 14 GB，实际 {result} GB"

        print("✅ ZeRO 显存函数测试通过！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_zero_memory()
```

### 练习 2.2: 最大模型规模反推

给定 GPU 显存容量和 ZeRO stage，反推最大可训练参数量。

```python
def max_trainable_params(gpu_memory_gb, num_gpus, zero_stage, overhead_ratio=0.2, model_dtype='fp16', optimizer='adam'):
    available_memory = gpu_memory_gb * (1 - overhead_ratio)
    model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
    gradient_bytes = model_bytes
    optimizer_bytes = {'adam': 12, 'sgd': 4}[optimizer]

    if zero_stage == 0 or zero_stage == 'ddp':
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes
    elif zero_stage == 1:
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes / num_gpus
    elif zero_stage == 2:
        bytes_per_param = model_bytes + gradient_bytes / num_gpus + optimizer_bytes / num_gpus
    elif zero_stage == 3:
        bytes_per_param = (model_bytes + gradient_bytes + optimizer_bytes) / num_gpus
    else:
        raise ValueError('zero_stage must be 0/ddp, 1, 2 or 3')

    return available_memory / bytes_per_param
```


```python
def test_max_trainable_params():
    try:
        result = max_trainable_params(80, 8, 'ddp', 0.2)
        assert abs(result - 4) < 1e-9, f"错误：DDP 应该最多训练 4B，实际 {result}B"

        result = max_trainable_params(80, 8, 3, 0.2)
        assert abs(result - 32) < 1e-9, f"错误：ZeRO-3 应该最多训练 32B，实际 {result}B"

        print("✅ 最大模型反推函数测试通过！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_max_trainable_params()
```

---
## Part 3: 场景应用

把 DDP、ZeRO-1、ZeRO-2、ZeRO-3 放在同一个表里，比较不同策略的可训练模型规模。

```python
gpu_memory = 80
num_gpus = 8
overhead_ratio = 0.2
strategies = [
    ('DDP', 'ddp'),
    ('ZeRO-1', 1),
    ('ZeRO-2', 2),
    ('ZeRO-3', 3),
]

print('8 x A100 80GB 的最大可训练模型规模（FP16 + Adam，预留 20% 显存）：')
print('-' * 78)
for name, stage in strategies:
    max_params = max_trainable_params(gpu_memory, num_gpus, stage, overhead_ratio)
    print(f"{name:<8} {max_params:>8.2f}B 参数")
```

### 练习 3.2: 梯度累积视角

梯度累积本身不减少单步显存，但会影响全局 batch size 的设置。你可以在此基础上继续扩展一个 `calculate_accumulation_steps` 函数。
---

## 🎓 总结

通过本练习，你应该能把 `DDP → ZeRO → 显存上限` 这一条链路串起来。

**下一步：** 学习 [07. CPU GPU Heterogeneous Scheduling](./07_CPU_GPU_Heterogeneous_Scheduling.md)，继续理解训练时的数据搬运与调度问题。
---

🛑 **STOP HERE** 🛑

> 请先自己完成上面的计算和测试，再看答案。