# 29. CUDA Stream Advanced Scheduling | CUDA Stream 高级调度

**难度：** Hard | **标签：** `CUDA`, `Stream`, `异步调度` | **目标人群：** 想把推理和训练流程调得更细的学习者

这一页是在基础 Stream 概念之上的进一步延伸。重点不是“什么是 Stream”，而是“怎么让多个任务更合理地并行、同步、回放和控制优先级”。

前置阅读建议先看 `1D-01 CPU/GPU 异构调度`，先把 Host / Device、PCIe 传输和基础异步概念建立起来，再看这页会更顺。

## Q1：CUDA Stream 为什么能做并行调度？

<details>
<summary>点击展开查看解析</summary>

CUDA Stream 可以看成一条命令队列：
- 同一条 Stream 内部的操作顺序执行
- 不同 Stream 之间在硬件允许时可以并行

这允许我们把任务拆成：
- 数据搬运
- 计算
- 后处理

只要这些阶段之间依赖关系允许，就可以把它们放到不同 Stream 里，从而提高设备利用率。

一个数量级直觉表可以帮助你判断“为什么要调度”：

| 操作 | 典型耗时 | 说明 |
| --- | --- | --- |
| 空 kernel launch | `5-10 μs` | CPU 下发指令的开销 |
| 小算子计算 | `1-2 μs` | GPU 实际计算很短 |
| 单 kernel 总耗时 | `6-12 μs` | launch 往往占主导 |
| CUDA Graph 回放 | `<1 μs` | 几乎不再重复 launch |

所以，Stream / Graph 的价值并不是“让 GPU 变魔法一样更强”，而是尽量少浪费在调度和启动上。
</details>

## Q2：CUDA Event 为什么重要？

<details>
<summary>点击展开查看解析</summary>

Event 的作用是跨 Stream 做同步点。

你可以把它理解成：
- 某个 Stream 先完成一段工作
- 记录一个 Event
- 另一个 Stream 在等待这个 Event 后继续执行

这比“全局阻塞”更精细，因为它只同步真正有依赖关系的部分。

所以在复杂流水线里，Event 往往是把异步调度真正串起来的关键。

Event 的开销通常很小，但它不是“零成本”；工程上常见的用法是只在真正有依赖关系的地方插入 Event，避免把同步点铺得太密。
</details>

## Q3：CUDA Graph 和 Stream 调度是什么关系？

<details>
<summary>点击展开查看解析</summary>

CUDA Graph 更像是把一整段稳定的执行路径捕获下来，再在后续回放。

它的价值在于：
- 降低频繁 kernel launch 的开销
- 减少 Python 或调度层的干预
- 在推理场景中稳定化执行路径

但它也有局限：
- 输入 shape 如果经常变化，捕获和回放就不稳定
- 依赖关系复杂时，图捕获也更难管理

所以 Graph 常和 Stream 一起出现，但它解决的是“固定流程回放”，不是替代所有异步调度。

什么时候值得用 Graph？
- 固定 batch size + 固定 seq length 的离线批处理推理
- 多次重复执行相同的计算模式，例如固定层数的 Transformer

什么时候不太划算？
- 变长请求很频繁的在线推理
- 动态 shape 经常变化、每次路径都不同的任务

可以把它记成一句话：Graph 适合“路径稳定”，不适合“形状经常跳”。
</details>

## Q4：Stream 优先级和典型流水线怎么理解？

<details>
<summary>点击展开查看解析</summary>

CUDA 支持高 / 低优先级 Stream。高优先级 Stream 中的 kernel 会更容易被优先调度，适合放高优请求或更敏感的控制流。

一个常见的三段流水线思路是：

```text
Stream A: H2D
Stream B: Kernel
Stream C: D2H + 后处理
```

更实用的做法不是把所有任务都丢进同一条 Stream，而是把职责拆开：
- 一个 Stream 负责输入搬运
- 一个 Stream 负责核心计算
- 一个 Stream 负责输出回传和收尾

再用 Event 连接依赖关系，这样更容易让搬运和计算重叠。
</details>

## Q5：高级调度最常见的误区是什么？

<details>
<summary>点击展开查看解析</summary>

- **“Stream 越多越好”**  
  不对。太多 Stream 会增加调度复杂度。

- **“异步一定更快”**  
  不对。只有当传输和计算能够重叠时，异步才更有意义。

- **“Event 就是锁”**  
  不准确。Event 是同步点，不是全局锁。

- **“CUDA Graph 适合所有推理”**  
  不对。动态 shape 很多时，Graph 的收益会下降。
</details>

## 小结

这一页的核心判断是：

> **Stream 解决并行调度，Event 解决局部同步，Graph 解决稳定路径回放。**

## 关联阅读

这一页和 1D-01 的异构调度、1D-02 的 CUDA / Triton 编程模型是连着的，建议一起看：

- `1D-01` CPU/GPU 异构调度
- `1D-02` CUDA/Triton 编程模型
- `1D-04` 动态 Shape 处理

## 配合练习

这一页建议和后面的 [练习页](./29_CUDA_Stream_Advanced_Scheduling_Practice.md) 一起看。这页的练习可以围绕下面三步展开：

- 先做一个 Stream 优先级的小实验，看高 / 低优先级任务的调度差异
- 再做 Event 同步，观察不同 Stream 之间的依赖是怎么被串起来的
- 最后尝试 CUDA Graph 捕获和回放，比较捕获与回放的耗时差异

对应的 Notebook 练习可以看这里：

- [练习页](./29_CUDA_Stream_Advanced_Scheduling_Practice.md)
