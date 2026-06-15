# 22. MoE Parameter and Compute Practice | 1A-04. MoE Parameter and Compute - 计算练习

**难度：** Medium-Hard | **标签：** `MoE`, `参数量`, `路由`, `系统直觉` | **目标人群：** 希望理解“大模型为什么能又大又省算”的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本练习配套理论文档：[22_MoE_Parameter_and_Compute.md](./22_MoE_Parameter_and_Compute.md)

这页不追求把 MoE 写成论文综述，而是把最重要的工程问题做成可验证练习：
- 总参数量怎么算
- 每个 token 真正激活多少参数
- Router / 容量限制 / 通信成本为什么会成为瓶颈

## 本节如何和 Notebook 配合

这一节建议和理论文档一起学：

- 先看理论页，建立“总容量大、激活计算量不一定大”的直觉
- 再做 Notebook，把参数量、负载均衡和通信成本真正算出来
- Notebook 里的测试用来确认你不是“看懂了”，而是真的“会算了”

### 练习目标

- 计算 dense FFN、MoE 专家部分和活跃参数量
- 用一个固定 token 分布模拟 Router 的负载不均衡
- 估算专家并行时的通信量，建立系统级直觉


```python
import math
from typing import Dict, List


def dense_ffn_params(d: int, d_ff: int) -> int:
    """Dense FFN/MLP 的参数量，按 2 * d * d_ff 估算。"""
    return 2 * d * d_ff


def moe_expert_params(d: int, d_ff: int, num_experts: int) -> int:
    """MoE 所有专家的总参数量。"""
    return num_experts * dense_ffn_params(d, d_ff)


def active_expert_params(d: int, d_ff: int, active_experts: int) -> int:
    """单个 token 激活的专家参数量。"""
    return active_experts * dense_ffn_params(d, d_ff)


def moe_layer_params(d: int, d_ff: int, num_experts: int, include_router: bool = True) -> int:
    """MoE 单层粗略参数量：Attention + Experts + Router。"""
    attention_params = 4 * d * d
    router_params = d * num_experts if include_router else 0
    return attention_params + moe_expert_params(d, d_ff, num_experts) + router_params


def to_million(x: int) -> float:
    return x / 1e6


def to_billion(x: int) -> float:
    return x / 1e9

```

## Part 1: 参数量与活跃参数量

### 练习 1.1: 计算 dense FFN 和 MoE 专家参数量

先把最基础的参数量公式写出来，再对比 dense 和 MoE 的差异。


```python
def test_moe_parameter_formulas():
    d = 4096
    d_ff = 16384
    experts = 8
    active = 2

    dense = dense_ffn_params(d, d_ff)
    total_expert = moe_expert_params(d, d_ff, experts)
    active_params = active_expert_params(d, d_ff, active)
    layer_params = moe_layer_params(d, d_ff, experts)

    assert dense == 134_217_728, dense
    assert total_expert == 1_073_741_824, total_expert
    assert active_params == 268_435_456, active_params
    assert layer_params == 1_140_883_456, layer_params
    print('✅ MoE 参数量公式测试通过')


test_moe_parameter_formulas()

print('单个 dense FFN 参数量:', to_million(dense_ffn_params(4096, 16384)), 'M')
print('8 个专家的总参数量:', to_billion(moe_expert_params(4096, 16384, 8)), 'B')
print('top-2 激活参数量:', to_billion(active_expert_params(4096, 16384, 2)), 'B')
print('MoE 单层粗略参数量:', to_billion(moe_layer_params(4096, 16384, 8)), 'B')

```

### 练习 1.2: 观察 MoE 的“总容量 vs 激活容量”

把总参数量和每次真正激活的参数量放在一起看，建立直觉。


```python
d = 4096
d_ff = 16384
experts = 8
active = 2

print('Mixtral 风格的数量级直觉：')
print('-' * 60)
print(f'dense FFN / expert: {to_million(dense_ffn_params(d, d_ff)):.1f} M params')
print(f'8 experts total   : {to_billion(moe_expert_params(d, d_ff, experts)):.2f} B params')
print(f'top-{active} active : {to_billion(active_expert_params(d, d_ff, active)):.2f} B params')
print(f'active ratio      : {active / experts:.0%}')
print(f'MoE layer rough   : {to_billion(moe_layer_params(d, d_ff, experts)):.2f} B params')

```

## Part 2: Router 与负载均衡

### 练习 2.1: 估算专家容量

Router 不只是决定 token 去哪里，还决定每个专家能不能“装得下”。


```python
def expert_capacity(num_tokens: int, num_experts: int, capacity_factor: float = 1.0) -> int:
    """每个专家可接收的 token 数量上限。"""
    return math.ceil(num_tokens / num_experts * capacity_factor)


def overflow_tokens(token_counts: List[int], capacity: int) -> int:
    """给定每个专家接收的 token 数量，计算总溢出 token 数。"""
    return sum(max(0, c - capacity) for c in token_counts)


def load_balance_stats(token_counts: List[int], capacity: int) -> Dict[str, float]:
    total = sum(token_counts)
    overflow = overflow_tokens(token_counts, capacity)
    return {
        'total_tokens': total,
        'capacity_per_expert': capacity,
        'max_tokens': max(token_counts),
        'min_tokens': min(token_counts),
        'overflow_tokens': overflow,
        'drop_rate': overflow / total if total else 0.0,
    }

```

### 练习 2.2: 观察不均衡路由

下面用一个固定 token 分布模拟 Router 负载不均衡，看看会发生什么。


```python
token_counts = [140, 40, 70, 90, 110, 80, 60, 110]
capacity = expert_capacity(sum(token_counts), len(token_counts), capacity_factor=1.0)
stats = load_balance_stats(token_counts, capacity)

print('专家 token 分布:', token_counts)
print('每个专家容量上限:', capacity)
print('-' * 60)
for k, v in stats.items():
    if k.endswith('rate'):
        print(f'{k:20s}: {v:.1%}')
    else:
        print(f'{k:20s}: {v}')

```


```python
def test_router_balance():
    token_counts = [140, 40, 70, 90, 110, 80, 60, 110]
    capacity = expert_capacity(sum(token_counts), len(token_counts), capacity_factor=1.0)
    stats = load_balance_stats(token_counts, capacity)

    assert capacity == 88, capacity
    assert stats['overflow_tokens'] == 98, stats['overflow_tokens']
    assert 0 < stats['drop_rate'] < 1
    print('✅ Router / 容量测试通过')


test_router_balance()

```

## Part 3: 通信成本与系统复杂度

### 练习 3.1: 粗估专家并行通信量

MoE 的工程代价不只在算术计算，还在专家并行带来的调度和通信。


```python
def estimate_expert_parallel_traffic(
    batch: int,
    seq_len: int,
    hidden_dim: int,
    num_experts: int,
    num_devices: int,
    bytes_per_param: int = 2,
    dispatch_and_gather_factor: float = 2.0,
) -> int:
    """粗略估算专家并行的通信量（字节）。"""
    tokens = batch * seq_len
    traffic = tokens * hidden_dim * bytes_per_param
    traffic *= (num_experts / max(1, num_devices))
    traffic *= dispatch_and_gather_factor
    return int(traffic)


def bytes_to_mb(x: int) -> float:
    return x / 1e6


def bytes_to_gb(x: int) -> float:
    return x / 1e9

```


```python
batch = 8
seq_len = 2048
hidden_dim = 4096
num_experts = 8
num_devices = 8
traffic = estimate_expert_parallel_traffic(batch, seq_len, hidden_dim, num_experts, num_devices)

print('专家并行通信量粗估：')
print('-' * 60)
print(f'batch={batch}, seq_len={seq_len}, hidden_dim={hidden_dim}')
print(f'communication = {bytes_to_mb(traffic):.1f} MB / layer')
print(f'communication = {bytes_to_gb(traffic):.3f} GB / layer')

```


```python
def test_traffic_estimation():
    traffic_small = estimate_expert_parallel_traffic(4, 1024, 4096, 8, 8)
    traffic_large = estimate_expert_parallel_traffic(8, 2048, 4096, 8, 8)
    assert traffic_small > 0
    assert traffic_large > traffic_small
    assert traffic_large == 4 * traffic_small
    print('✅ 通信量测试通过')


test_traffic_estimation()

```

### 练习 3.2: 思考训练与推理的差别

- 训练时更关心负载均衡、通信和收敛稳定性
- 推理时更关心 top-k 路由、显存占用和专家加载方式

如果要继续优化，你可以把这三件事一起看：
- active parameters
- router balance
- communication traffic

---

🛑 **STOP HERE** 🛑

先确保你能独立完成上面的练习，再看参考答案。

## 参考代码与解析

下面给出一版更精简的参考实现，帮助你对照自己写的版本。


```python
# ==================== 参考答案 ====================

def dense_ffn_params_ref(d: int, d_ff: int) -> int:
    return 2 * d * d_ff


def moe_expert_params_ref(d: int, d_ff: int, num_experts: int) -> int:
    return num_experts * dense_ffn_params_ref(d, d_ff)


def active_expert_params_ref(d: int, d_ff: int, active_experts: int) -> int:
    return active_experts * dense_ffn_params_ref(d, d_ff)


def expert_capacity_ref(num_tokens: int, num_experts: int, capacity_factor: float = 1.0) -> int:
    return math.ceil(num_tokens / num_experts * capacity_factor)

print('参考结论：')
print('- MoE 的关键不是总参数，而是激活参数和路由质量')
print('- Router 会把系统复杂度从算术计算转移到调度与通信')
print('- 训练和推理的瓶颈关注点不同')

```

### 解析

**1. 总参数 vs 活跃参数**

- dense FFN 和 MoE 专家部分都可以用 `2 × d × d_ff` 来估算
- MoE 的关键区别在于：总容量乘上专家数，但每个 token 只激活 top-k 个专家

**2. Router / 容量**

- 容量因子越小，越容易触发 token 溢出
- token 分布越不均衡，越容易出现部分专家过载

**3. 通信成本**

- 专家并行不是免费扩展
- batch、seq_len 和 hidden_dim 增大时，通信成本会很快上涨

## 关联阅读

MoE 的工程落地和 1C 的分布式通信章节强相关，建议后面接着看：

- `1C-01` 通信拓扑与互连技术
- `1C-03` 并行策略决策框架
- `1C-04` 通信调度优化

## 配合练习

这一页后面的练习可以按三步来设计：

- 先用一个简化公式算 Mixtral 8x7B 的总参数和活跃参数
- 再模拟 router 的负载均衡，看看 token 分配不均会带来什么后果
- 最后把专家并行的 All-to-All 通信量估出来，建立通信成本直觉
