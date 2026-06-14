# 14. FlashAttention Memory Model

**Difficulty:** Medium | **Tags:** `Attention`, `FlashAttention`, `Memory Model` | **Audience:** Learners preparing to enter Chapter 2

This page connects the Attention-and-memory intuition from Chapter 1 directly to FlashAttention in Chapter 2. When beginners first meet FlashAttention, they often think it is just "a faster Attention algorithm". The real point is not only speed: FlashAttention reorganizes intermediate computation and memory access. Once you understand that, you will understand why it became such an important part of modern LLM training and inference.

## Why this is a bridge page

- Chapter 1 already explained why Attention is memory-hungry
- Chapter 2 will move directly into Attention implementation, FlashAttention, and PagedAttention
- Without an understanding of intermediate matrices and memory-access cost, FlashAttention looks like a minor implementation trick
- This page exists to explain what it is actually saving

## The intuition you should build first

### 1. Standard Attention is not just expensive to compute

Standard Attention has two layers of cost:
- the computation itself is heavy
- the intermediate matrices consume a lot of memory

So the problem is not just "slow computation"; it is also "too much state to keep after computing".

### 2. FlashAttention's core idea is to reorganize the work

You can remember FlashAttention like this:

```text
split the big matrix into smaller tiles, process them in faster on-chip storage, and avoid unnecessary intermediate memory traffic
```

It is not just a new formula. It is a new way of moving and processing data.

### 3. Memory access patterns decide real performance

On GPUs, the bottleneck is often not only compute:
- repeated access to slow memory
- too many intermediate results
- data being moved back and forth unnecessarily

FlashAttention matters because it makes computation more aligned with the hardware.

## A common understanding path

You can think about FlashAttention in this order:

```text
standard Attention -> large intermediate matrices -> high memory pressure -> tiled processing -> less intermediate traffic -> better performance
```

This mental model is more important than memorizing implementation details.  
The Chapter 2 code follows this intuition.

## Common mistakes

- thinking FlashAttention is just "a faster softmax"
- focusing on the math form while ignoring memory access
- mixing up FlashAttention and PagedAttention
- assuming it solves all inference-memory problems, when it mainly targets Attention intermediates and access patterns

## After studying this page, you should be able to answer

- why standard Attention is memory-heavy
- what FlashAttention actually saves
- why tiled processing helps performance
- why Chapter 2 Attention topics should be studied together

## Connection to later chapters

- **Chapter 2: Attention MHA/GQA**  
  You will see how the attention computation itself is organized

- **Chapter 2: FlashAttention**  
  You will see the implementation ideas behind tiling, online softmax, and memory-friendly execution

- **Chapter 2: vLLM PagedAttention**  
  You will see that KV cache organization also affects inference memory, beyond Attention itself

## Summary

This page has one job:
- explain why standard Attention is expensive
- explain why FlashAttention works
- explain why Chapter 2 needs a dedicated FlashAttention page

If you can connect "compute form" and "memory access pattern", you already have the key intuition needed to enter the Chapter 2 Attention section.

