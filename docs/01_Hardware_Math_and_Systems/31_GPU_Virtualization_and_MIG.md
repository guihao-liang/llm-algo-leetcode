# 31. GPU Virtualization and MIG | GPU 虚拟化与 MIG

**难度：** Medium | **环境：** CPU-first | **标签：** `MIG`, `GPU 虚拟化` | **目标人群：** 多租户推理学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/31_GPU_Virtualization_and_MIG.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页讲的是一张 GPU 同时服务多个任务时，底层资源为什么必须被切分和隔离。

**关键词：** `MIG`, `MPS`, `multi-tenancy`
## 前置阅读

**导语：** 这一页先把资源隔离、动态 shape 和多租户调度的关系接上，再看为什么 GPU 虚拟化必须先切边界。

- [07. CPU and GPU Heterogeneous Scheduling | CPU 与 GPU 异构调度](./07_CPU_GPU_Heterogeneous_Scheduling.md)
- [29. CUDA Stream Advanced Scheduling | CUDA Stream 高级调度](./29_CUDA_Stream_Advanced_Scheduling.md)
- [30. Dynamic Shape Handling | 动态 Shape 处理](./30_Dynamic_Shape_Handling.md)

## 相关阅读

**导语：** 如果想继续把隔离、编译和成本判断串起来，可以接着看这些页。

- [19. Operator Fusion Introduction | 算子融合导论](./19_Operator_Fusion_Introduction.md)
- [32. TVM / MLIR Deep Practice | TVM / MLIR 深度实践](./32_TVM_MLIR_Deep_Practice.md)
- [33. TCO and Cost Model | 算力评估与 TCO 模型](./33_TCO_and_Cost_Model.md)

## Q1：GPU 虚拟化为什么首先是资源隔离问题？

<details><summary>点击展开查看解析</summary>

多租户场景里，GPU 不只是“给谁用”，而是“怎么让很多任务共用同一块卡，同时还不互相干扰”。

如果没有隔离，某个任务可能会抢占过多算力、缓存、显存或调度窗口，导致别的任务延迟飙升、尾延迟变差。

所以 GPU 虚拟化的第一目标，不是简单提高利用率，而是先把共享资源的干扰边界定义清楚，让性能更可控。
</details>
### Q1小验证：隔离先解决什么

先把“谁会干扰谁”看清楚，再谈共享效率。

```python
def isolate(total_gpu, slices):
    # 隔离不是一句 partitioned，而是看资源切片后每份能拿到多少边界。
    per_slice = total_gpu / slices
    fragmentation = total_gpu % slices
    return {'per_slice': round(per_slice, 2), 'fragmentation': fragmentation}

for case in [(1, 2), (1, 3), (1, 7)]:
    print(case, '->', isolate(*case))
print('isolation only makes sense when the slice boundaries are explicit')

```

## Q2：MIG 的本质是在切什么边界？

<details><summary>点击展开查看解析</summary>

MIG 的核心，不是“把一块 GPU 拆成几块”这么简单，而是把算力、显存和部分调度边界切开，让不同租户在更稳定的资源切片里运行。

这样做的直接收益，是减少缓存和算力争用带来的抖动；间接收益，是让推理服务更容易做容量规划和 QoS 控制。
</details>
### Q2小验证：切分后有什么收益

资源边界越清晰，推理尾延迟通常越稳定。

```python
def mig_benefit(stable_latency=True, less_contention=True, queue_depth=1):
    # MIG 的收益不是一个布尔值，而是尾延迟和争用一起下降多少。
    latency_gain = 2 if stable_latency else 0
    contention_gain = 2 if less_contention else 0
    queue_penalty = max(queue_depth - 1, 0)
    return latency_gain + contention_gain - queue_penalty

for case in [(True, True, 1), (True, False, 2), (False, True, 3)]:
    print(case, '-> benefit score', mig_benefit(*case))
print('MIG helps when tail latency improvement beats queue overhead')

```

## Q3：为什么隔离更强，吞吐未必更高？

<details><summary>点击展开查看解析</summary>

隔离做得越强，任务之间越不容易互相干扰，但共享的效率通常也会下降；共享做得越激进，利用率可能更高，但延迟和抖动也更容易恶化。

所以虚拟化问题的核心，不是追求“切得更细”本身，而是在隔离、吞吐和尾延迟之间找到能接受的折中。
</details>
### Q3小验证：为什么不能只看利用率

高利用率不一定意味着好的服务体验。

```python
def balance(isolation, throughput, jitter=0):
    # 隔离和吞吐不是二选一，而是看能接受多少抖动换多少共享效率。
    score = 0
    score += 2 if isolation else 0
    score += 2 if throughput else 0
    score -= jitter
    return score

plans = [
    ('strong_isolation', True, False, 0),
    ('strong_throughput', False, True, 2),
    ('balanced', True, True, 1),
]
for name, isolation, throughput, jitter in plans:
    print(name, '->', balance(isolation, throughput, jitter))
print('the best plan depends on how much jitter you can tolerate')

```

## ⚠️ 常见误区

- 虚拟化不是把 GPU 拆碎就结束了。
- MIG 关注的是隔离和可预测性，不只是拆分数量。
- 多租户推理里，延迟抖动常常比平均吞吐更重要。
- 共享和隔离本身就是一组折中。