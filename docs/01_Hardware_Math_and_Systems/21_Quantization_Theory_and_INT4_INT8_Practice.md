# 21. Quantization Theory and INT4 INT8 Practice | 1A-03. Quantization Theory and INT4/INT8 - 计算练习

**难度：** Medium | **标签：** `量化`, `显存计算`, `推理加速` | **目标人群：** 入门 / 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本练习配套理论文档：[21_Quantization_Theory_and_INT4_INT8.md](./21_Quantization_Theory_and_INT4_INT8.md)

建议先阅读理论文档，再来完成本练习。理论页负责建立量化直觉，Notebook 负责把显存、误差和速度差异真正算出来。

---

## 🎯 学习目标

- 理解 FP16 / INT8 / INT4 的显存差异
- 能够手动实现基础的 per-tensor 量化与反量化
- 学会比较量化前后的误差与推理收益
- 能够判断什么时候需要 PTQ、QAT 或更保守的方案
## 本节如何和 Notebook 配合

这一节建议和理论文档一起学：

- 先看理论页，建立量化收益 / 代价、权重 / 激活差异和 PTQ / QAT 的直觉
- 再做 Notebook，把显存、误差和速度差异真正算出来
- Notebook 里的测试用来确认你不是“看懂了”，而是真的“会算了”

### 练习目标

- 计算 FP16 / BF16 / INT8 / INT4 的权重显存占用
- 实现最简单的 per-tensor 量化与反量化
- 比较不同位宽下的量化误差与显存节省

## 题目区

这一部分先完成练习，再看答案。

- 先把每个函数自己写出来
- 再运行测试确认结果
- 如果卡住，先回看理论页的直觉部分

### TODO 概览

- TODO 1：计算 FP16 / INT8 / INT4 显存
- TODO 2：实现 per-tensor 量化与反量化
- TODO 3：比较不同位宽的误差与显存节省

## Part 1: 显存与位宽

### 练习 1.1: 实现模型权重显存计算函数

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

### 练习 1.2: 对比不同数据格式

使用上面的函数，对比 7B 模型在不同数据格式下的权重显存占用。

```python
# TODO: 计算 7B 模型在不同格式下的显存占用
num_params = 7
dtypes = ['fp16', 'bf16', 'int8', 'int4']

print('7B 模型权重显存占用对比：')
print('-' * 40)
for dtype in dtypes:
    memory = calculate_weight_memory(num_params, dtype)
    print(f'{dtype.upper():<6} {memory:>6.1f} GB')
```

---

## Part 2: 基础量化实现

### 练习 2.1: 实现 per-tensor 对称量化与反量化

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

### 练习 2.2: 观察量化误差

比较原始张量和反量化张量的误差。

```python
# TODO: 构造一个随机张量，观察 8-bit 和 4-bit 的误差差异
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

---

## Part 3: 量化收益与风险

### 练习 3.1: 比较显存节省与误差

把显存节省和误差放在一起看，判断量化是否值得。

```python
# TODO: 对比 7B 模型在 FP16 / INT8 / INT4 下的显存节省
base = calculate_weight_memory(7, 'fp16')
for dtype in ['int8', 'int4']:
    mem = calculate_weight_memory(7, dtype)
    saving = 1 - mem / base
    print(f'{dtype.upper():<4} memory={mem:.1f} GB, saving={saving:.0%}')
```

### 练习 3.2: 何时需要更保守的方案

思考：当误差过大、激活离群值明显或任务对精度敏感时，应该如何调整量化策略？

```python
# TODO: 写出 2-3 个需要保守量化或 QAT 的场景
scenarios = [
    '小模型生成任务对精度非常敏感',
    '激活值离群值明显，INT4 误差过大',
    '需要保持与基线几乎一致的输出质量'
]
for s in scenarios:
    print('-', s)
```

---

## Part 4: 自检与扩展

### 练习 4.1: 小结检查

如果你能回答下面三个问题，就说明这一页基本过关：

1. FP16 / INT8 / INT4 的显存差异是多少？
2. 为什么激活值通常比权重更难量化？
3. 什么时候应该考虑 QAT 而不是继续硬压位宽？

```python
def check_understanding():
    questions = [
        '1. FP16 / INT8 / INT4 的显存差异是多少？',
        '2. 为什么激活值通常比权重更难量化？',
        '3. 什么时候应该考虑 QAT 而不是继续硬压位宽？',
    ]
    for q in questions:
        print(q)

check_understanding()
```

---

🛑 **STOP HERE** 🛑

先确保你能独立完成上面的练习，再看参考答案。
## 参考答案与解析

> 下面的内容在正式讲解时再补全。当前版本先保留练习结构，方便后续统一填充答案。