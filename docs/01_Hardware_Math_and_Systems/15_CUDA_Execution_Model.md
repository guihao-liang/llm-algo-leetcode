# 15. CUDA Execution Model

**Difficulty:** Medium | **Tags:** `CUDA`, `Kernel`, `Execution Model` | **Audience:** Learners preparing to enter Chapter 3

This page connects Chapter 1's hardware understanding to Chapter 3's kernel implementation work. You already know why GPUs are fast, why memory is expensive, and why Attention becomes a bottleneck. Now you need to understand how a GPU actually launches and runs a kernel. Once you have that execution model, CUDA, Triton, shared memory, and streams will stop feeling like isolated syntax.

## Why this is a bridge page

- Chapter 1 already covered GPU architecture, communication, memory, and programming-model background
- Chapter 3 will move directly into Triton basics, CUDA kernels, shared memory, streams, and operator fusion
- If you do not understand threads, blocks, grids, and warps, the implementation chapters are hard to follow
- So this page exists to answer one question first: how does a GPU execute a kernel?

## The intuition you should build first

### 1. A GPU is not "one thread running"

CUDA execution is usually organized around:
- grid
- block
- warp
- thread

You do not need every detail here, but you should know:
- a thread is the smallest execution unit
- a warp is the basic hardware scheduling granularity
- a block is the unit of shared-memory cooperation
- a grid is the full kernel launch layout

### 2. Kernel performance depends heavily on layout

The same matrix operation can perform very differently depending on:
- how work is split
- how data is tiled
- how shared memory is used
- whether global-memory access is contiguous

These decisions directly affect the kernels you will see in Chapter 3.

### 3. Asynchrony and overlap matter

Many GPU optimizations are not just "compute faster":
- overlap computation with data movement
- schedule work better across streams
- reduce waiting between kernels

That is why CUDA streams and scheduling matter later.

## A simple mental model

You can think of a CUDA kernel like this:

```text
split the big job into many tiles -> assign them to blocks -> split each block into threads -> execute threads in warps
```

This is not a full formula, but it is enough to reason about:
- which code styles are more GPU-friendly
- which layouts lead to contiguous access
- which designs make shared-memory reuse easier

## Common mistakes

- treating CUDA as "just multi-threading"
- focusing on syntax while ignoring memory access and thread organization
- assuming kernel speed only depends on code length
- thinking streams are automatic speedups without understanding scheduling

## After studying this page, you should be able to answer

- why CUDA kernels are organized by block / warp / thread
- why tiling matters for performance
- why shared memory and contiguous access are so important
- why Chapter 3 needs execution-model knowledge, not just syntax

## Connection to later chapters

- **Chapter 3: Triton Fundamentals**  
  You will see how Triton maps programs to block-like execution units

- **Chapter 3: CUDA Kernel and Memory Optimization**  
  You will see how warp behavior, shared memory, streams, and scheduling affect performance

- **Chapter 3: Operator Fusion**  
  You will see how kernel layout decides whether multiple ops can be fused efficiently

## Summary

This page has one job:
- explain how GPU kernels are organized
- explain why memory access and block structure matter
- explain why Chapter 3 needs these concepts before code

If you can connect "thread organization" with "memory access", you have the basic intuition needed to enter Chapter 3.

