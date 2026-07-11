# 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化

**难度：** Hard | **环境：** CPU-first | **标签：** `算力评估`, `ZeRO` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页把 DDP、ZeRO 和激活值显存的理论推导落成可运行代码，帮助你从“会算”走到“会验证”。

**关键词：** `VRAM`, `ZeRO`, `AdamW`

## 前置阅读

**导语：** 这一页先把混合精度训练和显存计算的底层推导接上，再进入 DDP / ZeRO 的具体公式。

- [05. Communication Topologies | 通信拓扑与分布式基石](./05_Communication_Topologies.md)
- [03. GPU Architecture and Memory | GPU 物理架构与内存层级](./03_GPU_Architecture_and_Memory.md)
- [Group 1C: Distributed Communication and Memory Sharing | 1C: 多卡通信与显存共享](./1C.md)

## 相关阅读

**导语：** 如果想继续把显存估算和训练策略补完整，可以接着看这些页。

- [21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](./21_Quantization_Theory_and_INT4_INT8.md)
- [26. Parallel Strategy Decision Framework | 并行策略决策框架](./26_Parallel_Strategy_Decision_Framework.md)
- [29. Tensor Parallelism Sim | Tensor 并行模拟](../02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.md)

## Q1：DDP 显存计算

<details>
<summary>点击展开查看解析</summary>

在 DDP 下，每张卡都保存完整模型、完整梯度和完整优化器状态。对于 Adam + FP16/BF16 训练，常用的粗略估算是 **16Φ**：

- 模型参数：`2Φ`
- 梯度：`2Φ`
- 优化器状态：`12Φ`

把这三部分相加，就得到 `16Φ` 这一条经验公式。

</details>
### Q1小验证：DDP 显存计算

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

## Q2：ZeRO 显存计算

<details>
<summary>点击展开查看解析</summary>

ZeRO 的核心思想是把训练状态分摊到多张 GPU 上：

- **ZeRO-1**：切分优化器状态
- **ZeRO-2**：切分优化器状态和梯度
- **ZeRO-3**：切分参数、梯度和优化器状态

因此，随着 stage 提升，单卡显存会持续下降，但通信和调度复杂度会增加。

</details>
### Q2小验证：ZeRO 显存计算

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

### Q2扩展验证：最大模型规模反推

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

## Q3：8×80GB GPU 下的最大模型规模估算

**问题：** 如果你手上只有 8 张 80GB GPU，并且希望预留 20% 显存给 activation 和通信开销，DDP、ZeRO-1、ZeRO-2、ZeRO-3 分别能支撑多大的模型？

请把四种策略放在同一个表里比较最大可训练模型规模。

<details>
<summary>点击展开查看解析</summary>

把 DDP、ZeRO-1、ZeRO-2、ZeRO-3 放在同一个表里，比较不同策略的可训练模型规模。这个问题的目标是把前面的 DDP / ZeRO 公式真正落到工程场景里，而不是只停留在概念层。

</details>

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
