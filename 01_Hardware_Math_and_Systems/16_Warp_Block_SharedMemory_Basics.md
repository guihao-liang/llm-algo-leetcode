# 16. Warp、Block 与 Shared Memory 基础 | Warp, Block and Shared Memory Basics

**难度：** Medium | **标签：** `CUDA`, `Warp`, `Block`, `Shared Memory` | **目标人群：** 准备进入 Chapter 3 的学习者

这一页把 Chapter 1 的 GPU 直觉，接到 Chapter 3 的线程组织方式上。重点是 warp、block 和 shared memory 怎么配合。

## 前置关系

- Chapter 3 会直接进入 Triton 和 CUDA kernel 的实现
- 先理解 warp / block / shared memory，后面的代码才好看懂

## 你应该先建立的直觉

### 1. warp 是硬件执行的基本粒度

在 GPU 上，线程不是完全独立执行的，很多时候是按 warp 这一级一起调度。

你只需要先记住：
- warp 决定了线程协同执行的方式
- 数据访问如果不整齐，warp 的效率会受影响
- 后面看 kernel 性能时，warp 不是“附属概念”，而是核心概念

### 2. block 是协作和共享的边界

block 内的线程可以更方便地协作，也更适合共享局部数据。

这意味着：
- block 的划分方式会影响并行度
- block 的大小会影响共享内存使用
- 合理的 block 组织是写好 kernel 的第一步

### 3. shared memory 是加速的关键资源

shared memory 很快，但容量有限，所以它的价值在于：
- 缓存重复使用的数据
- 减少对慢速全局内存的访问
- 配合分块计算提高吞吐

这也是为什么很多 GPU 优化都会围绕 shared memory 展开。

## 一个最常见的理解路径

```text
大问题切成小块 -> 分给 block -> block 内按 warp 和 thread 协作 -> 热数据放进 shared memory
```

这条线比记住每个术语的定义更重要。  
后面 Chapter 3 的 kernel 和 Triton 实现，基本都建立在这个直觉上。

## 常见误区

- 把 warp、block、thread 当成完全独立的概念
- 只看线程数量，不看数据访问模式
- 以为 shared memory 越多越好
- 把 GPU 编程理解成“把 CPU 线程模型搬过去”

## 这一页学完后，你应该能回答

- warp、block、thread 的关系是什么
- 为什么 block 划分会影响 kernel 性能
- 为什么 shared memory 是重要的加速资源
- 为什么 Chapter 3 的优化要先看执行层次

## 和后续章节的联系

- **Chapter 3: Triton 基础**  
  你会看到 Triton 如何把 block 级别的思维表达出来

- **Chapter 3: CUDA 内核优化**  
  你会看到 shared memory、warp 协作和访问模式如何影响性能

- **Chapter 3: 算子融合**  
  你会看到更好的线程组织如何为融合创造条件

## 小结

这一页的作用很简单：
- 先让你知道 GPU 线程是怎么协作的
- 再让你知道 shared memory 为什么重要
- 最后让你知道为什么 Chapter 3 要从这些基础开始

如果你能把 warp、block 和 shared memory 串起来理解，这一页的目标基本达成。
