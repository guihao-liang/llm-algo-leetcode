# 24. SRAM Optimization Techniques | SRAM 优化技术

**难度：** Hard | **环境：** GPU optional | **标签：** `SRAM`, `Shared Memory`, `Optimization` | **目标人群：** kernel 优化入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页讲的是为什么 shared memory 很快，但也很容易写出效果不好的 kernel。

**关键词：** `shared memory`, `tile`, `bank conflict`
## 前置阅读

**导语：** 这一页先把 shared memory 的复用、访问模式和寄存器预算接上，再看为什么片上优化常常不是“能不能用”，而是“能不能用对”。

- [15. CUDA Execution Model | CUDA 执行模型](./15_CUDA_Execution_Model.md)
- [16. Warp Block SharedMemory Basics | Warp、Block 与 Shared Memory 基础](./16_Warp_Block_SharedMemory_Basics.md)
- [23. TensorCore Deep Dive | Tensor Core 深度剖析](./23_TensorCore_Deep_Dive.md)

## 相关阅读

**导语：** 如果想继续把片上优化和更高层的 kernel 组织、稀疏和编译优化串起来，可以接着看这些页。

- [08. Programming Models and CUDA/Triton | 编程模型演进](./08_Programming_Models_CUDA_Triton.md)
- [18. Triton Block Model | Triton Block 模型](./18_Triton_Block_Model.md)
- [25. Sparse Computation and Sparse Attention | 稀疏计算与稀疏注意力](./25_Sparse_Computation_and_Sparse_Attention.md)

## Q1：shared memory 的收益为什么必须和同步代价一起看？

<details><summary>点击展开查看解析</summary>

shared memory 的优势不是“本身很快”，而是“它能不能用复用把 HBM 访问替换掉”。

如果数据只被读一次，或者为了共享付出的同步次数太多，片上访问的收益就会被抵消。

所以判断 shared memory 值不值得用，第一件事不是看容量，而是看“复用收益是否大于同步和搬运代价”。
</details>
### Q1小验证：复用够不够

先算复用带来的收益，能不能覆盖同步和搬运。

```python
def sram_gain(reuse_times, sync_points=0, hbm_cost=10, smem_cost=2, sync_cost=3):
    # shared memory 不是无条件更快，复用要足够高才值得支付同步代价。
    reuse_benefit = max(reuse_times - 1, 0) * (hbm_cost - smem_cost)
    penalty = sync_points * sync_cost
    return {'net_gain': reuse_benefit - penalty, 'worth_it': reuse_benefit > penalty}

for case in [(1, 0), (2, 1), (5, 1), (5, 3)]:
    print(case, '->', sram_gain(*case))
print('shared memory pays off only when reuse amortizes synchronization and搬运')

```

## Q2：bank conflict 为什么会把本来并行的访问串行化？

<details><summary>点击展开查看解析</summary>

bank conflict 的本质，不是“读写慢”这么简单，而是多个线程在同一个时钟周期里想访问同一个 bank，硬件只能分轮次服务。

因此，看起来并行的访问会被拆开，访问轮次增加，吞吐自然下降。

这也是为什么 shared memory 优化不只看容量，更要看 stride、对齐方式和访问分布。
</details>
### Q2小验证：访问模式为什么重要

先看 stride 和 bank 是否会把并行访问打成串行。

```python
def bank_conflict_penalty(stride, banks=32):
    # stride 不是越大越慢，关键是是否把多个线程打到同一个 bank。
    conflict = (stride % banks == 0)
    partial = (stride % 2 == 0) and not conflict
    return {'conflict': conflict, 'penalty': 4 if conflict else 2 if partial else 1}

for stride in [1, 2, 16, 32, 33]:
    print(stride, '->', bank_conflict_penalty(stride))
print('conflict means parallel accesses are serialized across banks')

```

## Q3：tile、布局和占用率为什么要一起设计？

<details><summary>点击展开查看解析</summary>

tile 决定块怎么切，布局决定这些块能不能连续读到数据，占用率决定你有没有足够的线程把这些 tile 真正跑满。

如果只调 tile，不管布局和占用率，shared memory 的收益很容易卡在访存不连续或者 block 资源不够上。

所以这里不是三个独立选项，而是一个共同决定“能不能把片上缓存吃满”的组合问题。
</details>
### Q3小验证：切块和布局是不是绑在一起

先把 tile、layout 和占用率当成一个组合来考虑。

```python
def layout_tile_score(tile, contiguous=True, occupancy=1.0, reuse=1):
    # tile、layout 和 occupancy 一起决定 shared memory 能吃到多少收益。
    if not contiguous:
        return {'score': 0, 'reason': 'layout_fragmented'}
    score = tile * reuse * occupancy
    return {'score': round(score, 2), 'reason': 'good' if occupancy > 0.5 else 'low_occupancy'}

plans = [(64, True, 0.8, 1), (128, True, 0.9, 3), (128, False, 0.9, 3), (256, True, 0.4, 4)]
for plan in plans:
    print(plan, '->', layout_tile_score(*plan))
print('tile, layout and occupancy are a joint design problem')

```

## Q4：为什么寄存器预算会决定片上优化的上限？

<details><summary>点击展开查看解析</summary>

当寄存器压力过高时，编译器会把一部分临时值 spill 到更慢的层级。

这会让你前面为了减少 HBM 访问、提高复用、扩大 tile 做的优化，重新付出额外的访存和同步代价；如果 spill 太多，甚至会把 occupancy 也一起拉低。

所以片上优化不是只看 shared memory 是否够快，而是要同时看寄存器预算、spill 风险和 block 资源是否还能维持在可接受区间。
</details>
### Q4小验证：为什么 spill 很危险

一旦 spill，局部复用就不再只看片上路径。

```python
def spill_tradeoff(registers_needed, register_budget=64, reuse_gain=0, occupancy=1.0):
    # 寄存器预算一旦超了，spill 会把片上优化的收益吃回去。
    overflow = max(registers_needed - register_budget, 0)
    spill_penalty = overflow * 2
    reuse_bonus = reuse_gain * 3
    occupancy_penalty = round(max(1.0 - occupancy, 0) * 4, 2)
    net = reuse_bonus - spill_penalty - occupancy_penalty
    return {'net_gain': round(net, 2), 'spill': overflow > 0, 'occupancy_penalty': occupancy_penalty}

plans = [
    ('safe_tiling', 40, 64, 1, 0.9),
    ('reuse_rich', 64, 64, 3, 0.8),
    ('spill_risk', 80, 64, 4, 0.8),
    ('spill_heavy', 96, 64, 5, 0.5),
]
for name, regs, budget, reuse, occ in plans:
    print(name, '->', spill_tradeoff(regs, budget, reuse, occ))
print('when spill or low occupancy dominates, shared-memory gains get eaten away')

```
