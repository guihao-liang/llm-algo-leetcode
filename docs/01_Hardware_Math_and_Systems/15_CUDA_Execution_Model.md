# 15. CUDA Execution Model | CUDA Execution Model

**难度：** Medium | **标签：** `CUDA`, `Kernel`, `Execution Model` | **目标人群：** 准备进入 Chapter 3 的学习者

这一页把 Chapter 1 的硬件理解，接到 Chapter 3 的内核实现上。重点是先理解 GPU 怎么把 kernel 跑起来。

## 为什么这一节是前置页

- Chapter 1 前面已经讲了 GPU 架构、通信、显存和编程模型的背景
- Chapter 3 会直接进入 Triton 基础、CUDA kernel、shared memory、stream 和算子融合
- 如果读者还不知道线程、block、grid、warp 之间是什么关系，后面的实现会很难跟上
- 所以这一节的目标是：先把“GPU 是怎么调度 kernel 的”讲明白

## 你应该先建立的直觉

### 1. GPU 不是“一个线程在跑”，而是“大量线程按层级组织起来跑”

CUDA 的执行模型里，最常见的层级是：
- grid
- block
- warp
- thread

你不需要在这一页记住所有细节，但要知道：
- thread 是最小执行单元
- warp 是硬件调度的基本粒度
- block 是共享内存和协作的边界
- grid 是整个 kernel 的总体任务分配

### 2. kernel 的性能和组织方式强相关

同样是一个矩阵运算，写法不同，性能可能差很多：
- 线程怎么分配
- 数据怎么切块
- 共享内存怎么用
- global memory 访问是否连续

这些都直接影响后面 Chapter 3 里的 kernel 质量。

### 3. 异步执行和计算/通信重叠很重要

GPU 的很多优化，不只是“算得更快”，而是：
- 让计算和数据搬运重叠
- 让不同 stream 的任务更合理地并行
- 让 kernel 之间减少等待

这会直接影响你后面看 CUDA Stream、Triton 调度和通信优化的理解。

## 一个最常见的判断方式

你可以把一个 CUDA kernel 粗略想成：

```text
把一个大问题切成很多小块 -> 分配给多个 block -> 每个 block 再分给多个 thread -> 线程通过 warp 方式执行
```

这不是完整公式，但足够让你在 Chapter 3 里判断：
- 哪种写法更像 GPU 友好的切块
- 哪种数据布局更容易形成连续访问
- 哪种设计更适合 shared memory 复用

## 常见误区

- 把 CUDA 当成“只是多线程编程”
- 只关注语法，不关注线程组织和内存访问
- 认为 kernel 快慢只和代码行数有关
- 把 stream 当成“自动加速”，而不理解它的调度意义

## 这一页学完后，你应该能回答

- 为什么 GPU kernel 要按 block / warp / thread 组织
- 为什么数据切块对性能影响很大
- 为什么 shared memory 和连续访问这么重要
- 为什么 Chapter 3 不能只看代码语法，而要理解执行模型

## 和后续章节的联系

- **Chapter 3: Triton 基础**  
  你会看到 Triton 如何用更高层的方式表达 block / program 的映射

- **Chapter 3: CUDA 内核与显存优化**  
  你会看到 warp、shared memory、stream 和 kernel 调度如何真正影响性能

- **Chapter 3: 算子融合**  
  你会看到 kernel 组织方式如何决定多个算子能否高效融合

## 小结

这一页的作用也很简单：
- 先让你知道 GPU kernel 是怎么组织起来的
- 再让你知道为什么 memory access 和 block 结构重要
- 最后让你知道为什么 Chapter 3 要把这些概念落到实现里

如果你能把“线程组织”和“内存访问”联系起来，这一页的目标基本达成。
