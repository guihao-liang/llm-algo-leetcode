# 01. Data Types and Precision | 大模型的数据格式与混合精度

**难度：** Easy | **环境：** CPU-first | **标签：** `基础概念`, `混合精度` | **目标人群：** 通用基础 (算法/Infra)

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*



在计算任何大模型的显存或算力之前，先把“数据”在 GPU 中的表示方式搞清楚。这是所有硬件推导和量化算法的基础。本节我们将从最基础的字节换算开始，结合工业界常见的 A100 与较前沿的 H100 架构，一路走到混合精度的底层逻辑和 FP8 的设计思路。

**关键词：** `FP16`, `BF16`, `INT8`

## 前置阅读
**导语：** 这一页要先把张量、自动求导、调试和性能意识接上，后面的数据格式、显存估算和量化推导才不会只停留在公式层。

- [Group 0B: PyTorch Tensors and Autograd | 0B: PyTorch 张量与自动求导](../00_Prerequisites/0B.md)
- [Group 0E: Debugging and Performance | 0E: 调试与性能](../00_Prerequisites/0E.md)
- [02. LLM Params and FLOPs | 大模型参数量与算力推导](./02_LLM_Params_and_FLOPs.md)

## 相关阅读
**导语：** 如果想把“数据格式 -> 参数量 -> 显存 -> 量化”这条线补完整，可以按这个顺序继续看：
- [02. LLM Params and FLOPs | 大模型参数量与算力推导](./02_LLM_Params_and_FLOPs.md)：先把参数量和算力推导接起来。
- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](./06_VRAM_Calculation_and_ZeRO.md)：再看训练 / 推理场景下的显存拆分与 ZeRO 思路。
- [21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](./21_Quantization_Theory_and_INT4_INT8.md)：最后补量化理论和 INT4 / INT8 的视角。

## Q1：基础认知——常见的数据格式分别占用多大内存空间？

<details>
<summary>点击展开查看解析</summary>

在计算机底层，1 Byte（字节）= 8 bits（位）。大模型中常见的格式占用如下：

- **FP32 (单精度浮点数)**: 32 bits = **4 Bytes**
- **FP16 (半精度浮点数)**: 16 bits = **2 Bytes**
- **BF16 (BFloat16)**: 16 bits = **2 Bytes**
- **INT8 (8位整型)**: 8 bits = **1 Byte**
- **INT4 (4位整型)**: 4 bits = **0.5 Byte** (通常用于极度压缩的量化如 AWQ/GPTQ)

**实战估算：**
做静态显存估算时，只需要把“模型参数量”乘以对应的“字节数”即可。比如一个 7B（70亿）参数的模型，如果采用 FP16/BF16 加载，纯权重占用的显存就是：$7 \times 10^9 \times 2 \text{ Bytes} \approx 14 \text{ GB}$。
</details>
### Q1小验证：基础显存计算

实现一个函数，计算给定参数量和数据格式的模型显存占用。


```python
def calculate_model_memory(num_params_b, dtype):
    """
    计算模型参数的显存占用
    
    Args:
        num_params_b: 参数量（单位：B，即十亿）
        dtype: 数据类型，可选 'fp32', 'fp16', 'bf16', 'int8', 'int4'
    
    Returns:
        memory_gb: 显存占用（单位：GB）
    
    示例:
        >>> calculate_model_memory(7, 'fp16')
        14.0
        >>> calculate_model_memory(7, 'int8')
        7.0
    """
    # 每种数据类型占用的字节数
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    memory_gb = num_params_b * bytes_per_param[dtype]
    return memory_gb
```


```python
# 测试函数
def test_calculate_model_memory():
    try:
        # 测试用例 1: LLaMA-7B FP16
        result = calculate_model_memory(7, 'fp16')
        assert result == 14, f"错误：LLaMA-7B FP16 应该是 14 GB，实际 {result} GB"
        
        # 测试用例 2: LLaMA-7B INT8
        result = calculate_model_memory(7, 'int8')
        assert result == 7, f"错误：LLaMA-7B INT8 应该是 7 GB，实际 {result} GB"
        
        # 测试用例 3: LLaMA-13B FP16
        result = calculate_model_memory(13, 'fp16')
        assert result == 26, f"错误：LLaMA-13B FP16 应该是 26 GB，实际 {result} GB"
        
        # 测试用例 4: LLaMA-70B INT4
        result = calculate_model_memory(70, 'int4')
        assert result == 35, f"错误：LLaMA-70B INT4 应该是 35 GB，实际 {result} GB"
        
        print("✅ 所有测试通过！")
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_model_memory()
```

### Q1扩展验证：对比不同数据格式

使用上面的函数，对比 LLaMA-7B 在不同数据格式下的显存占用。


```python
# 对比 LLaMA-7B 在不同格式下的显存占用
model_name = "LLaMA-7B"
num_params = 7
dtypes = ['fp32', 'fp16', 'bf16', 'int8', 'int4']

print(f"{model_name} 显存占用对比：")
print("-" * 40)
for dtype in dtypes:
    memory = calculate_model_memory(num_params, dtype)
    print(f"{dtype.upper():<8} {memory:>6.1f} GB")
```

## Q2：底层原理——同样是 16-bit，FP16 和 BF16 的底层位分布有什么本质区别？

<details>
<summary>点击展开查看解析</summary>

这涉及浮点数在底层的位分布设计：一个浮点数由 **符号位 (Sign)** + **指数位 (Exponent)** + **尾数位/精度位 (Mantissa/Fraction)** 组成。
核心法则是：**指数位决定了数值的范围大小，尾数位决定了数值的精确度。**

1. **FP16 的结构**：1 位符号 + **5 位指数** + 10 位尾数。
   - 5 位指数意味着它能表示的最大数值只有 **65504**。
   - 尾数长，所以它对小数部分的表示非常“精细”。

2. **BF16 (Brain Float 16) 的结构**：1 位符号 + **8 位指数** + 7 位尾数。
   - 它是 Google Brain 专门为深度学习发明的。它其实就是直接把 FP32（8位指数）砍掉了后面的 16 位尾数！
   - 因为拥有 8 位指数，BF16 能表示的最大数值范围和 FP32 一模一样（高达 $3.4 \times 10^{38}$），**极难发生数值溢出**。代价是尾数位从 10 降到了 7，牺牲了一点数值的“精确度”。
</details>

### Q2小验证：混合精度训练显存计算

在混合精度训练中，显存占用包括：
- 模型参数（FP16/BF16）：2Φ
- 梯度（FP16/BF16）：2Φ
- 优化器状态（FP32）：
  - FP32 主权重：4Φ
  - 一阶动量（Adam）：4Φ
  - 二阶动量（Adam）：4Φ
  - 总计：12Φ

**总显存 = 2Φ + 2Φ + 12Φ = 16Φ**


```python
def calculate_training_memory(num_params_b, model_dtype='fp16', optimizer='adam'):
    """
    计算混合精度训练的总显存占用
    
    Args:
        num_params_b: 参数量（单位：B）
        model_dtype: 模型数据类型（'fp16' 或 'bf16'）
        optimizer: 优化器类型（'adam' 或 'sgd'）
    
    Returns:
        total_memory_gb: 总显存占用（单位：GB）
    
    示例:
        >>> calculate_training_memory(7, 'fp16', 'adam')
        112.0
    """
    model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
    gradient_bytes = model_bytes
    optimizer_bytes = 12 if optimizer == 'adam' else 4
    total_memory_gb = num_params_b * (model_bytes + gradient_bytes + optimizer_bytes)
    return total_memory_gb
```


```python
# 测试函数
def test_calculate_training_memory():
    try:
        # 测试用例 1: LLaMA-7B + Adam
        result = calculate_training_memory(7, 'fp16', 'adam')
        assert result == 112, f"错误：应该是 112 GB，实际 {result} GB"
        
        # 测试用例 2: LLaMA-7B + SGD
        result = calculate_training_memory(7, 'fp16', 'sgd')
        assert result == 56, f"错误：应该是 56 GB，实际 {result} GB"
        
        # 测试用例 3: LLaMA-13B + Adam
        result = calculate_training_memory(13, 'bf16', 'adam')
        assert result == 208, f"错误：应该是 208 GB，实际 {result} GB"
        
        print("✅ 所有测试通过！")
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_training_memory()
```


```python
# 分析 LLaMA-7B 训练时的显存分布
num_params = 7

model_params = num_params * 2  # FP16 模型参数
gradients = num_params * 2     # FP16 梯度
optimizer_states = num_params * 12  # FP32 优化器状态
total = model_params + gradients + optimizer_states

print(f"LLaMA-7B 混合精度训练显存分布：")
print("-" * 50)
print(f"模型参数 (FP16):      {model_params:>6.1f} GB ({model_params/total*100:>5.1f}%)")
print(f"梯度 (FP16):          {gradients:>6.1f} GB ({gradients/total*100:>5.1f}%)")
print(f"优化器状态 (FP32):    {optimizer_states:>6.1f} GB ({optimizer_states/total*100:>5.1f}%)")
print("-" * 50)
print(f"总计:                 {total:>6.1f} GB")
print("\n结论：优化器状态占据了大部分显存！")
```

## Q3：训练实战——为什么现代大模型预训练（如 LLaMA）大多转向了 BF16，而不再首选 FP16？

<details>
<summary>点击展开查看解析</summary>

这一转变主要源于大语言模型深层 Transformer 结构固有的数值不稳定性，以及硬件生态的演进。

**1. FP16 的核心挑战：动态范围受限，上溢风险高**

FP16 的动态范围（最大值约 65504）远窄于 FP32（约 $3.4 \times 10^{38}$）。在大模型训练中：

- **上溢问题（主要痛点）**：Attention 的 logits、未归一化的激活值、或某些层的梯度容易超过 65504，造成上溢变成 `NaN`，导致训练崩溃。这是 FP16 训练的**最大障碍**。
  
- **下溢问题（次要）**：反向传播中的小梯度（如 $10^{-7}$ 量级）可能因精度不足被截断，影响参数更新。

**Loss Scaling 的权宜之计**：

为了使用 FP16 训练，必须引入 **Loss Scaling（损失缩放）**：
- **原理**：在反向传播前放大损失值（如乘以 1024），将小梯度放大到 FP16 可表示范围，计算完成后再缩小回来，从而缓解下溢。
- **局限**：只能解决梯度的下溢，无法解决前向传播中的上溢；且缩放因子需要动态调整（检测溢出后减小，长期无溢出时增大），增加了工程复杂度。
- **历史成功**：早期的 BERT、GPT-2 等模型在 V100（仅支持 FP16 Tensor Core）上通过精心调优仍实现了稳定训练。

**2. BF16 的三大优势：稳定、简单、硬件支持**

- **极大的动态范围**：BF16 继承了 FP32 的 8 位指数，最大值达 $3.4 \times 10^{38}$，**在实际训练中几乎不会上溢**——这是相比 FP16 的决定性优势。
  
- **免调参的稳定性**：无需 Loss Scaling 等额外技巧，直接替换 FP32 即可稳定训练，大幅降低工程复杂度。
  
- **硬件原生支持**：从 A100（Ampere 架构）开始，NVIDIA 在 Tensor Core 中原生支持 BF16，性能与 FP16 相当，使 BF16 从”实验性格式”变成”工业标准”。

**3. 精度 vs 范围：为什么神经网络更需要范围？**

虽然 BF16 的尾数位从 10 位降到 7 位（损失约 3 位精度），但：

- **神经网络对精度损失鲁棒**：训练是长期累积的统计过程，单步的微小舍入误差会被后续更新”平滑”掉。
- **上溢是灾难性的**：一旦出现 `NaN`，会立即传播到整个模型，导致训练不可恢复地崩溃。

因此，用较小的精度代价换取极大的数值稳定性，BF16 成为了现代大模型预训练（如 LLaMA、GPT-3、PaLM）的主流选择。

**4. 数值对比与应用场景**

| 格式 | 指数位 | 尾数位 | 最大值 | 训练稳定性 | 主要应用 |
|------|--------|--------|--------|-----------|---------|
| FP32 | 8 | 23 | $10^{38}$ | 最稳定 | 基准/调试 |
| FP16 | 5 | 10 | $6.5 \times 10^4$ | 需 Loss Scaling | 推理优化 |
| BF16 | 8 | 7 | $10^{38}$ | 极稳定 | **大模型训练** ✅ |

**补充说明**：FP16 并未被完全”抛弃”——在推理场景中，由于不涉及梯度计算，数值范围需求较小，FP16 的更高精度（10 位尾数）反而能带来更好的输出质量，许多推理框架（如 TensorRT、vLLM）仍优先使用 FP16。

**总结**：BF16 在训练中的主导地位源于其”大范围 + 硬件支持 + 零调参”的综合优势，完美契合了大模型训练对数值稳定性的需求。
</details>

### Q3小验证：量化显存节省计算


```python
def calculate_quantization_savings(num_params_b, from_dtype, to_dtype):
    """
    计算量化后的显存节省
    
    Args:
        num_params_b: 参数量（单位：B）
        from_dtype: 原始数据类型
        to_dtype: 量化后的数据类型
    
    Returns:
        savings_gb: 节省的显存（单位：GB）
        savings_percent: 节省的百分比
    
    示例:
        >>> calculate_quantization_savings(7, 'fp16', 'int8')
        (7.0, 50.0)
    """
    original_memory = calculate_model_memory(num_params_b, from_dtype)
    quantized_memory = calculate_model_memory(num_params_b, to_dtype)
    savings_gb = original_memory - quantized_memory
    savings_percent = savings_gb / original_memory * 100
    return savings_gb, savings_percent
```


```python
# 测试函数
def test_calculate_quantization_savings():
    try:
        # 测试用例 1: FP16 -> INT8
        savings_gb, savings_percent = calculate_quantization_savings(7, 'fp16', 'int8')
        assert savings_gb == 7, f"错误：应该节省 7 GB，实际 {savings_gb} GB"
        assert savings_percent == 50, f"错误：应该节省 50%，实际 {savings_percent}%"
        
        # 测试用例 2: FP16 -> INT4
        savings_gb, savings_percent = calculate_quantization_savings(7, 'fp16', 'int4')
        assert savings_gb == 10.5, f"错误：应该节省 10.5 GB，实际 {savings_gb} GB"
        assert savings_percent == 75, f"错误：应该节省 75%，实际 {savings_percent}%"
        
        print("✅ 所有测试通过！")
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_quantization_savings()
```

## Q4：系统进阶——在 BF16 混合精度训练 (AMP) 时，为什么优化器里依然必须保留一份 FP32 的主权重 (Master Weights)？

<details>
<summary>点击展开查看解析</summary>

在混合精度训练中，模型的前向传播（Forward）和反向传播（Backward）都是用 16-bit 跑的，这样可以节省 50% 的显存并利用 Tensor Core 加速。

但问题出在**参数更新（Optimizer Step）**这一步：
$$ W_{new} = W_{old} - \text{Learning\_Rate} \times \text{Gradient} $$

在大模型训练后期，学习率（LR）通常非常小（例如 $10^{-5}$），计算出的梯度通常也很小。两者的乘积是一个极其微小的更新量 $\Delta W$。
如果 $W_{old}$ 也是用 16-bit 格式保存的，由于 16-bit 的尾数位非常短（表示小数的精度低），它**根本无法识别**这种极其微小的累加。结果就是，这个极小的 $\Delta W$ 加上去之后，因为**精度不足发生下溢出 (Underflow)**，更新量被直接抹零（即类似于 $1.0 + 0.0000001$ 在 16-bit 下依然等于 $1.0$），导致模型彻底停止学习！

因此，为了保证模型能吃进最微小的梯度更新，优化器中必须始终保留一份 **FP32 的 Master Weights**。每次反向传播算出梯度后，都会在**高精度的 FP32 副本上**进行参数累加更新，然后再将其截断强转回 16-bit 格式，送给下一轮的前向计算。这就是为什么开启全参训练后，显存占用远大于静态权重的原因。
</details>
### Q4小验证：实际场景应用

给定 GPU 显存容量，计算能加载多大的模型。


```python
def max_model_size(gpu_memory_gb, dtype, overhead_ratio=0.2):
    """
    计算给定 GPU 显存能加载的最大模型参数量
    
    Args:
        gpu_memory_gb: GPU 显存容量（单位：GB）
        dtype: 数据类型
        overhead_ratio: 预留给 KV Cache 和激活值的显存比例（默认 20%）
    
    Returns:
        max_params_b: 最大参数量（单位：B）
    
    示例:
        >>> max_model_size(80, 'fp16', 0.2)
        32.0
    """
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5,
    }
    available_memory = gpu_memory_gb * (1 - overhead_ratio)
    max_params_b = available_memory / bytes_per_param[dtype]
    return max_params_b
```


```python
# 测试不同 GPU 能加载的最大模型
gpus = [
    ('RTX 3090', 24),
    ('RTX 4090', 24),
    ('A100 40GB', 40),
    ('A100 80GB', 80),
    ('H100 80GB', 80),
]

print("不同 GPU 能加载的最大模型参数量（FP16，预留 20% 显存）：")
print("-" * 60)
print(f"{'GPU':<15} {'显存':<10} {'最大模型 (FP16)':<20} {'最大模型 (INT8)'}")
print("-" * 60)

for gpu_name, memory in gpus:
    max_fp16 = max_model_size(memory, 'fp16', 0.2)
    max_int8 = max_model_size(memory, 'int8', 0.2)
    print(f"{gpu_name:<15} {memory:>4} GB     {max_fp16:>6.1f}B              {max_int8:>6.1f}B")
```

## Q5：硬件标杆——作为开启大语言模型时代的决定性硬件，A100（Ampere 架构）在数据精度支持上做出了哪些革命性升级？

<details>
<summary>点击展开查看解析</summary>

A100 之所以能成为大模型时代的重要工业界标杆（至今仍被广泛使用），其核心原因之一就是在硬件层面原生支持了两种很重要的混合精度格式，进一步释放了算力：

**1. 原生支持 BF16 Tensor Core**
- 在 A100 之前的 V100（Volta 架构）时期，Tensor Core **只支持 FP16** 乘法。这就是为什么早期研究人员在训练模型时深受溢出之苦。
- A100 在其第三代 Tensor Core 中直接加入了原生的 BF16 乘加支持。这使得算力直接翻倍（相比 FP32）的同时，一举解决了动态范围溢出的所有痛点，使得千亿参数大模型的稳定训练成为可能。

**2. 引入了神兵利器：TF32 (TensorFloat-32)**
- NVIDIA 为了让开发者在不改动任何祖传代码（继续写纯 FP32 代码）的情况下也能用上 Tensor Core 的加速，发明了 TF32 格式。
- TF32 是一种精妙的混合态格式：它拥有 **FP32 的指数位（8位，保证不溢出）** 和 **FP16 的尾数位（10位，保证精度）**，总共占用 19 个 bit 的信息，但在显存中依然按 32位 存储。
- **底层机制**：当你在 PyTorch 中设置 `torch.backends.cuda.matmul.allow_tf32 = True`（在 A100 及更新的架构上这是默认开启的）时，如果你向 GPU 丢了两个 FP32 矩阵相乘，A100 会在内部的 Tensor Core 里将其**截断为 TF32 更快算完**，然后再转回 FP32 输出。这让“看似是单精度”的矩阵乘法获得数倍级的性能提升。
</details>
### Q5小验证：A100 的精度路径

比较 A100 在 FP32 / TF32 / BF16 / FP16 下的计算路径和张量核心利用方式。


```python
def a100_precision_path(requested_dtype, allow_tf32=True):
    table = {
        'fp32': {'storage_bits': 32, 'tensor_core': allow_tf32, 'path': 'tf32' if allow_tf32 else 'fp32', 'speed_hint': 1.0 if not allow_tf32 else 4.0},
        'tf32': {'storage_bits': 32, 'tensor_core': True, 'path': 'tf32', 'speed_hint': 4.0},
        'bf16': {'storage_bits': 16, 'tensor_core': True, 'path': 'bf16', 'speed_hint': 2.0},
        'fp16': {'storage_bits': 16, 'tensor_core': True, 'path': 'fp16', 'speed_hint': 2.0},
    }
    return table[requested_dtype]

for dtype in ['fp32', 'tf32', 'bf16', 'fp16']:
    print(dtype, '->', a100_precision_path(dtype))
print('A100 makes BF16 and TF32 first-class paths for Tensor Core acceleration')

```

## Q6：前沿演进——NVIDIA 在 H100（Hopper 架构）中引入的原生 FP8 格式，有什么专门针对 AI 的设计？

<details>
<summary>点击展开查看解析</summary>

随着模型越来越大，16-bit 已经满足不了工业界的胃口了。NVIDIA 在 H100 中引入了极其激进的原生 FP8 计算。

只有 8 个 bit 空间，既想要保留一点表示数值大小的“范围”，又想要保留一点表示小数的“精度”，这太矛盾了！所以 NVIDIA 极其聪明地把 FP8 强行分成了**两种独立的变体**，用于深度学习的不同生命周期：

1. **E4M3 格式** (4位指数 + 3位尾数)：
   - **侧重：精度**。
   - **用途**：专用于**前向传播 (Forward) 和激活值 (Activations)**。因为前向计算对“精度”更敏感，且产生的激活数值通常分布得比较集中，不需要太大的范围（指数）。
2. **E5M2 格式** (5位指数 + 2位尾数)：
   - **侧重：动态范围**。
   - **用途**：专用于**反向传播 (Backward) 和梯度 (Gradients)**。因为反向传播过程中的梯度数值跨度极大，非常容易出现下溢出，所以必须多分配 1 个位给指数，哪怕牺牲一点尾数精度也在所不惜。

这种将 8 个比特充分利用起来的软硬件协同设计，使得 H100 的 FP8 算力吞吐量相比 A100 的 FP16 还有明显提升，也为 8-bit 大模型训练和推理提供了基础。
</details>
### Q6小验证：H100 的 FP8 变体选择

对比 E4M3 和 E5M2，看看前向和反向为什么要分开设计。


```python
def fp8_variant_policy(stage):
    variants = {
        'forward': {'format': 'E4M3', 'exp_bits': 4, 'mantissa_bits': 3, 'focus': 'precision', 'stage': 'forward / activations'},
        'backward': {'format': 'E5M2', 'exp_bits': 5, 'mantissa_bits': 2, 'focus': 'range', 'stage': 'backward / gradients'},
    }
    return variants[stage]

for stage in ['forward', 'backward']:
    print(stage, '->', fp8_variant_policy(stage))
print('H100 splits FP8 into two variants to balance precision and dynamic range')

```
