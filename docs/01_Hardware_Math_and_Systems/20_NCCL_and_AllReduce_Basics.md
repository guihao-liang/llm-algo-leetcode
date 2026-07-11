# 20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础

**难度：** Medium | **环境：** CPU-first | **标签：** `NCCL`, `AllReduce`, `Distributed Training` | **目标人群：** 分布式训练入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页把多卡通信的底层直觉讲清楚，重点是知道 AllReduce 为什么重要、NCCL 为什么常被放在并行训练和分布式扩展里一起谈。

**关键词：** `NCCL`, `AllReduce`, `DP`
## 前置阅读

**导语：** 先把通信和并行层级对齐，再看 AllReduce 与 NCCL 会更顺。

- [Group 1C: Distributed Communication and Memory Sharing | 1C: 多卡通信与显存共享](./1C.md)
- [Group 1E: Compiler Optimization and Hardware Ecosystem | 1E: 编译优化与硬件生态](./1E.md)
- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](./13_Profiling_and_Bottleneck_Analysis.md)

## 相关阅读

**导语：** 把多卡通信放进 ZeRO、Pipeline、Tensor Parallelism 里看，能更好判断通信代价。

- [27. ZeRO Optimizer Sim | ZeRO 优化器模拟](../02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.md)
- [28. Pipeline Parallelism MicroBatch | Pipeline 并行微批次](../02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.md)
- [09. Triton PagedAttention | KV Cache 间接寻址](../03_Triton_Kernels/09_Triton_PagedAttention.md)

## Q1：NCCL 在分布式训练里到底解决什么问题？

<details>
<summary>点击展开查看解析</summary>

NCCL 解决的是多 GPU 之间高效通信的问题。

在分布式训练里，单卡算完还不够，参数、梯度和状态都要在多卡之间同步。如果通信效率太低，GPU 就会在等待数据时空转，扩展效果会很差。

NCCL 的作用，就是给这些通信原语提供高效实现，让多卡之间的同步尽量接近硬件能力上限。它本质上不是“额外功能”，而是分布式训练能否真正扩展的基础设施。
</details>
### Q1小验证：为什么多卡离不开通信库

先记住：多卡不是把计算复制几份就结束了。

```python
def step_time(compute_ms, sync_ms, tasks):
    # 多卡场景下，除了计算，还要付出同步成本。
    return compute_ms + sync_ms * max(tasks - 1, 0)

for tasks in [1, 2, 8]:
    print(tasks, 'GPUs ->', step_time(10, 3, tasks), 'ms/step')

```

## Q2：AllReduce 为什么是最核心的通信原语之一？

<details>
<summary>点击展开查看解析</summary>

AllReduce 的核心是“每张卡都有一份数据，最后大家要得到同一个聚合结果”。

这在数据并行训练里特别常见，因为每张卡都会计算局部梯度，最后需要把梯度做求和或平均，再同步回所有设备。这个过程如果效率低，就会拖慢整条训练链。

AllReduce 之所以重要，是因为它几乎代表了“多卡协同”的基础动作。很多通信优化、拓扑设计和同步策略，本质上都在围绕它展开。
</details>
### Q2小验证：为什么梯度同步离不开 AllReduce

先把“局部结果 -> 全局一致”这件事想清楚。

```python
def allreduce_mean(values):
    total = sum(values)
    return [total / len(values)] * len(values)

gradients = [0.8, 1.2, 1.0, 1.4]
print('before:', gradients)
print('after :', allreduce_mean(gradients))

```

## Q3：为什么通信带宽和拓扑会直接影响训练扩展效果？

<details>
<summary>点击展开查看解析</summary>

通信不是只有“能不能通”的问题，更重要的是“通得快不快”。

如果带宽低、拓扑绕路多、同步链路长，那么多卡之间的通信时间就会迅速放大，最后训练速度可能并不会随着 GPU 数量线性提升。

所以判断并行方案时，不能只看算力堆了多少卡，还要看通信是否会成为主瓶颈。拓扑、带宽、同步原语和调度策略共同决定了最终的扩展效率。
</details>
### Q3小验证：为什么拓扑会影响速度

先判断是带宽问题，还是路由和同步问题。

```python
def comm_time_ms(size_mb, bandwidth_gbps):
    # 粗略估算：MB -> Mb，再除以 Gbps。
    return size_mb * 8 / bandwidth_gbps

payload_mb = 256
cases = {'nvlink': 900, 'pcie': 64}
for name, bw in cases.items():
    print(name, '->', round(comm_time_ms(payload_mb, bw), 2), 'ms')
print('slowdown:', round(comm_time_ms(payload_mb, 64) / comm_time_ms(payload_mb, 900), 1), 'x')

```

## Q4：Ring AllReduce 为什么常被用来解释 AllReduce 的实现思路？

<details><summary>点击展开查看解析</summary>

Ring AllReduce 的关键点，不是“名字里有 ring”，而是它把一次全量聚合拆成了两个更可控的阶段：

1. **Reduce-Scatter**：先沿着环把各卡的数据逐步归约，并把结果切成若干块分散到各卡；
2. **All-Gather**：再沿着环把这些局部归约结果重新拼回完整结果。

这样做的好处是：每张卡在每一轮只发送和接收一小块数据，通信负载更均匀，不需要某一张卡一次性承担全部传输压力。

所以 NCCL 之所以重要，不只是因为它“能做 AllReduce”，而是因为它能把这些通信原语落成对拓扑更友好的实现。对于训练来说，真正的瓶颈常常不是有没有通信，而是通信是否能被拆成适合硬件链路的节奏。
</details>
### Q4小验证：AllReduce 为什么常拆成两段

```python
def ring_rounds(world_size):
    # Ring AllReduce 通常可拆成两个阶段，每阶段有 world_size - 1 轮。
    return 2 * (world_size - 1)

for n in [2, 4, 8]:
    print(n, 'ranks ->', ring_rounds(n), 'rounds')
print('phases:', ['reduce-scatter', 'all-gather'])

```

## ⚠️ 常见误区

- NCCL 不是模型结构的一部分，但它会直接决定多卡训练是否顺畅。
- AllReduce 不是唯一通信原语，但它是最常见、最关键的同步动作之一。
- 多卡扩展不是“卡越多越快”，通信常常会限制实际收益。
- 先看通信原语，再看拓扑和带宽，通常更容易定位分布式瓶颈。