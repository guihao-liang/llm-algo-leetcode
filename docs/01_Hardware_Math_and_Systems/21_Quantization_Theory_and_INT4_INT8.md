# 21. Quantization Theory and INT4 INT8 | 量化理论与 INT4/INT8

**难度：** Medium | **环境：** CPU-first | **标签：** `量化`, `推理优化` | **目标人群：** 量化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页不是要把量化算法讲成论文综述，而是要回答一个更实际的问题：为什么把权重从 FP16 压到 INT8 / INT4 后，模型能明显变小、推理也可能更快，但效果又不会“简单地等比例下降”。

**关键词：** `INT8`, `INT4`, `scale`, `zero point`, `PTQ`

## 前置

**导语：** 这一页先接上参数量、显存和 GPU 内存层级的判断，方便理解低比特量化为什么会同时影响存储、带宽和推理吞吐。

- [Part 1: 01. Data Types and Precision | 大模型的数据格式与混合精度](./01_Data_Types_and_Precision.md)
- [Part 1: 02. LLM Params and FLOPs | 大模型参数量与算力推导](./02_LLM_Params_and_FLOPs.md)
- [Part 1: 03. GPU Architecture and Memory | GPU 物理架构、内存层级与核心硬件单元](./03_GPU_Architecture_and_Memory.md)

## 相关阅读

**导语：** 如果还想继续看量化和工程决策的关系，可以再看显存计算、硬件选型和成本模型这几页。

- [Part 1: 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](./06_VRAM_Calculation_and_ZeRO.md)
- [Part 1: 10. AI Chips Overview and Alternatives | 算力现状与替代方案](./10_Domestic_AI_Chips_Overview.md)
- [Part 1: 33. TCO and Cost Model | 总拥有成本与成本模型](./33_TCO_and_Cost_Model.md)

## Q1：为什么量化能显著减少显存和带宽压力？

<details>
<summary>点击展开查看解析</summary>

如果把权重从 FP16 压到 INT8，单个参数的存储从 2 Bytes 变成 1 Byte，理论上权重显存约减半；如果压到 INT4，则理论上进一步降到 0.5 Byte，权重体积还会继续下降。

一个最直观的 7B 模型例子可以帮助建立数量级直觉：

| 权重格式 | 每参数字节数 | 7B 模型权重显存 | 相对 FP16 |
| --- | --- | --- | --- |
| FP16 | 2 Bytes | 14 GB | 1x |
| INT8 | 1 Byte | 7 GB | 0.5x |
| INT4 | 0.5 Byte | 3.5 GB | 0.25x |

这对推理很重要，原因不只是“模型更小了”，还因为：
- 显存占用下降后，更大的 batch 或更长上下文更容易装进去
- 读权重时需要搬运的数据更少，HBM 带宽压力也会下降
- 在一些带宽受限的场景里，吞吐会明显改善

但要注意，量化通常不会让所有成本都按比特数线性下降：
- 激活值可能仍然保留更高精度
- 部分层会保留 FP16 / BF16 累加
- 反量化和 scale 处理也有额外开销
- INT8 / INT4 是否真的加速，还取决于硬件是否原生支持低比特矩阵运算；如果硬件没有对应 Tensor Core / MMA 支持，量化有时只会省显存，不一定省时间

所以量化的收益通常是“显存更小 + 带宽更低 + 吞吐更高”，而不是单纯的“位宽缩小了多少，速度就提升多少”。
</details>
### Q1小验证：实现模型权重显存计算函数

实现一个函数，计算给定参数量和数据格式的模型权重显存占用。

```python
def calculate_weight_memory(num_params_b, dtype):
    """
    计算模型权重的显存占用。

    Args:
        num_params_b: 参数量（单位：B，即十亿）
        dtype: 数据类型，可选 'fp16', 'bf16', 'int8', 'int4'

    Returns:
        memory_gb: 显存占用（单位：GB）
    """
    bytes_per_param = {'fp16': 2, 'bf16': 2, 'int8': 1, 'int4': 0.5}[dtype]
    return num_params_b * bytes_per_param


def test_calculate_weight_memory():
    result = calculate_weight_memory(7, 'fp16')
    assert abs(result - 14.0) < 1e-9
    result = calculate_weight_memory(7, 'int8')
    assert abs(result - 7.0) < 1e-9
    result = calculate_weight_memory(7, 'int4')
    assert abs(result - 3.5) < 1e-9
    print('✅ calculate_weight_memory tests passed')

# 运行测试
test_calculate_weight_memory()
```

### Q1扩展验证：对比不同数据格式

使用上面的函数，对比 7B 模型在不同数据格式下的权重显存占用。

```python
# 直接计算 7B 模型在不同格式下的权重显存占用
num_params = 7
dtypes = ['fp16', 'bf16', 'int8', 'int4']

print('7B 模型权重显存占用对比：')
print('-' * 40)
for dtype in dtypes:
    memory = calculate_weight_memory(num_params, dtype)
    print(f'{dtype.upper():<6} {memory:>6.1f} GB')

```

## Q2：对称量化、非对称量化、per-tensor、per-channel 有什么区别？

<details>
<summary>点击展开查看解析</summary>

最常见的量化写法可以写成：

```text
q = round(x / scale) + zero_point
```

其中：
- `scale` 决定数值映射比例
- `zero_point` 决定零点是否偏移

常见组合有四种：
- **对称量化**：`zero_point = 0`，实现简单，常用于权重
- **非对称量化**：保留 `zero_point`，更适合分布偏移明显的数据
- **per-tensor**：整个张量共用一组 scale
- **per-channel**：每个通道单独一组 scale，通常精度更好，但元数据更多

经验上：
- 权重量化常常更适合 per-channel
- 激活量化常常更依赖校准数据
- 极低比特时，误差主要不来自平均值，而来自离群值和分布偏斜

为什么激活更难量化？
- 权重分布通常相对稳定，离群值更少
- 激活会随 token、层和上下文变化，分布波动更大
- 一些激活通道可能出现明显离群值，直接压到低比特时更容易失真
- 这也是为什么很多方案会做权重-激活协同处理，或者引入 SmoothQuant 这类预处理思路

这也是为什么真正落地时，量化不是“把 dtype 改小”这么简单，而是要同时决定 scale、zero point、分组粒度和累加精度。
</details>
### Q2小验证：实现 per-tensor 对称量化与反量化

实现一个最简单的量化函数，输入浮点张量，输出量化整数和 scale。

```python
import torch


def quantize_per_tensor(x, num_bits=8):
    """
    对张量做对称 per-tensor 量化。

    Returns:
        q: 量化后的整数张量
        scale: 量化比例
    """
    qmax = 2 ** (num_bits - 1) - 1
    scale = x.abs().max() / qmax if x.numel() > 0 else torch.tensor(1.0, device=x.device, dtype=x.dtype)
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax).to(torch.int8)
    return q, scale


def dequantize_per_tensor(q, scale):
    return q.to(torch.float32) * scale


def test_quantize_per_tensor():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q, scale = quantize_per_tensor(x, 8)
    x_hat = dequantize_per_tensor(q, scale)
    assert q.dtype == torch.int8
    assert x_hat.shape == x.shape
    print('q:', q.tolist())
    print('scale:', float(scale))
    print('x_hat:', x_hat.tolist())
    print('✅ quantize_per_tensor tests passed')

# 运行测试
test_quantize_per_tensor()
```

### Q2扩展验证：观察量化误差

比较原始张量和反量化张量的误差。

```python
# 直接观察 8-bit 和 4-bit 的量化误差差异
torch.manual_seed(0)
x = torch.randn(1024) * 2

for bits in [8, 4]:
    q, scale = quantize_per_tensor(x, bits)
    x_hat = dequantize_per_tensor(q, scale)
    mse = torch.mean((x - x_hat) ** 2).item()
    max_err = torch.max(torch.abs(x - x_hat)).item()
    print(f'{bits}-bit -> MSE={mse:.6f}, max_err={max_err:.6f}')

```


```python
def test_quantization_practice():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q8, s8 = quantize_per_tensor(x, 8)
    x8 = dequantize_per_tensor(q8, s8)
    q4, s4 = quantize_per_tensor(x, 4)
    x4 = dequantize_per_tensor(q4, s4)
    assert q8.dtype == torch.int8 and q4.dtype == torch.int8
    assert x8.shape == x.shape and x4.shape == x.shape
    assert torch.mean((x - x4).abs()) >= torch.mean((x - x8).abs())
    print('✅ 21 Quantization tests passed')

test_quantization_practice()
```

## Q3：PTQ、QAT、GPTQ、AWQ、GGUF 该怎么理解？

<details>
<summary>点击展开查看解析</summary>

这几个名字其实不在同一层：

- **PTQ / QAT** 先回答的是“量化发生在什么时候”
- **GPTQ / AWQ** 回答的是“训练后量化时，用什么方法尽量保住精度”
- **GGUF** 回答的是“量化结果最后怎么打包和分发”

从原理上看，可以按三层来理解：

### 1) PTQ 和 QAT：量化介入的位置
- **PTQ（Post-Training Quantization）**：先把模型训练完，再用少量校准数据估计 scale / zero point，把权重或激活映射到低比特空间。它的核心优点是成本低、落地快；缺点是量化误差没有在训练阶段被显式优化。
- **QAT（Quantization-Aware Training）**：在训练或微调时就把量化误差“模拟”进去，让模型参数学会适应低比特表示。它的核心思路不是单纯更精确，而是把误差提前暴露给优化过程，让模型自己补偿。

### 2) GPTQ 和 AWQ：训练后量化时，怎么减少误差
- **GPTQ** 更像是“带误差补偿的权重量化”。它利用少量校准数据近似估计二阶信息，量化某一层时尽量把量化误差压到对输出影响最小的方向上。直觉上，它不是只看每个权重的大小，而是看“哪些改动更伤输出”，然后做局部修正。
- **AWQ** 更强调“激活感知”。它会关注不同通道在真实输入下的重要性，尽量保护那些对输出更敏感的通道，让少数关键通道保留更高的表示质量。它的核心不是把所有权重平均压缩，而是先找出“最不能丢”的部分。

### 3) GGUF：量化结果如何被部署和加载
- **GGUF** 更偏文件格式和生态封装，不是单一的量化算法。它会把量化后的权重、scale、元数据和加载所需的信息组织成便于本地推理引擎读取的形式。
- 它的价值在于让量化模型更容易被 mmap、分发和跨工具链使用，所以更像“量化成果的交付格式”。

如果把它们放到同一张图里看：

- **PTQ / QAT**：决定“在哪一步量化”
- **GPTQ / AWQ**：决定“量化时怎么尽量保精度”
- **GGUF**：决定“量化完怎么存、怎么发、怎么加载”

所以不要把它们当成谁更先进的单选题，而要看你当前要解决的问题：
- 想快速落地，先看 PTQ
- 想在 PTQ 下尽量少掉点精度，看 GPTQ / AWQ
- 想把模型交付到本地推理生态，看 GGUF

如果你要判断什么时候更适合保守量化或直接考虑 QAT，可以先看这几个典型场景：

- 小模型生成任务对精度非常敏感
- 激活值离群值明显，INT4 误差过大
- 需要保持与基线几乎一致的输出质量

</details>
### Q3小验证：比较显存节省与误差

把显存节省和误差放在一起看，判断量化是否值得。

```python
# 直接对比 7B 模型在 FP16 / INT8 / INT4 下的显存节省
base = calculate_weight_memory(7, 'fp16')
for dtype in ['int8', 'int4']:
    mem = calculate_weight_memory(7, dtype)
    saving = 1 - mem / base
    print(f'{dtype.upper():<4} memory={mem:.1f} GB, saving={saving:.0%}')

```

## Q4：量化什么时候要考虑 QAT？

<details>
<summary>点击展开查看解析</summary>

QAT（Quantization-Aware Training）不是第一选择，但它在以下场景里很有价值：

- PTQ 之后精度损失过大，比如困惑度或任务指标下降明显
- 模型本身对低比特特别敏感，尤其是较小模型或生成类任务
- 你有足够的训练数据和算力，能够接受再训练或微调成本

一句话判断：
- 如果目标是“快速落地”，先用 PTQ
- 如果目标是“把低比特精度尽量拉回来”，再考虑 QAT

QAT 的代价是训练流程更复杂、成本更高，但它能把量化误差直接纳入训练过程，是 PTQ 之外的重要补救路线。
</details>
### Q4小验证：什么时候该考虑 QAT？

把量化误差和训练预算放在一起，判断是否需要从 PTQ 转向 QAT。


```python
def qat_recommendation(ptq_drop, acceptable_drop=0.5, retrain_budget_hours=0, sensitivity='medium'):
    score = 0
    if ptq_drop > acceptable_drop:
        score += 2
    if sensitivity == 'high':
        score += 1
    if retrain_budget_hours >= 10:
        score += 1
    if ptq_drop > acceptable_drop and retrain_budget_hours >= 10:
        recommendation = 'QAT'
    else:
        recommendation = 'PTQ'
    return {
        'recommendation': recommendation,
        'risk_score': score,
        'ptq_drop': ptq_drop,
        'acceptable_drop': acceptable_drop,
    }

cases = [
    (0.2, 0.5, 0, 'low'),
    (0.8, 0.5, 12, 'high'),
    (0.6, 0.5, 8, 'high'),
]
for case in cases:
    print(case, '->', qat_recommendation(*case))
print('QAT is worth considering when PTQ drop is too large and retraining budget exists')

```

## Q5：量化最常见的误区是什么？

<details>
<summary>点击展开查看解析</summary>

常见误区有四个：

- **“INT4 一定比 INT8 好”**  
  不对。比特更低并不自动更优，误差和硬件支持都可能让 INT4 更难用。

- **“量化只是改一下 dtype”**  
  不对。真正的量化会涉及校准、分组、反量化、累加精度和 kernel 支持。

- **“量化一定不影响效果”**  
  不对。不同层、不同通道、不同模型对低比特的容忍度差别很大。

- **“量化只影响权重”**  
  也不完整。推理时真正卡住性能的常常还包括激活、缓存和带宽。

这一页只要记住一句话：量化的目标不是“把精度尽可能压低”，而是在“误差可接受”的前提下把显存和带宽压力降下来。
</details>
### Q5小验证：量化配置里最常见的问题

看看哪些配置更容易踩坑，而不是把量化简单理解成“降 dtype”。


```python
def quantization_risk(bitwidth, hardware_support=True, calibration_quality=1.0, activation_sensitive=False):
    score = 0
    if bitwidth <= 4:
        score += 2
    if not hardware_support:
        score += 2
    if calibration_quality < 0.7:
        score += 1
    if activation_sensitive:
        score += 1
    if score >= 4:
        level = 'high'
    elif score >= 2:
        level = 'medium'
    else:
        level = 'low'
    return {
        'risk_level': level,
        'risk_score': score,
        'bitwidth': bitwidth,
    }

cases = [
    (8, True, 0.9, False),
    (4, True, 0.8, True),
    (4, False, 0.6, True),
]
for case in cases:
    print(case, '->', quantization_risk(*case))
print('quantization fails when bitwidth, calibration, and hardware support are all under pressure')

```
