# 11. KV Cache and Memory Growth

**Difficulty:** Medium | **Tags:** `Inference Memory`, `Attention`, `KV Cache` | **Audience:** Learners preparing to enter Chapter 2 / 3

This page is not here to re-teach Attention. Its job is to connect the Chapter 1 ability to estimate memory with the real inference problems you will meet in Chapter 2. Once a model starts generating long contexts, KV cache becomes the main source of memory growth. Understanding it is the difference between knowing a model can run and knowing whether it can keep running stably at scale.

## Why this is a bridge page

- Chapter 1 already covered parameter count, memory, and the basic cost of Attention
- Chapter 2 will move directly into `Attention MHA/GQA`, `FlashAttention`, and `vLLM PagedAttention`
- Without a clear KV cache model, those optimizations look like disconnected tricks
- This page exists to explain why inference memory grows in the first place

## The intuition you should build first

### 1. Training memory and inference memory are different

During training, memory is usually dominated by:
- parameters
- gradients
- optimizer states
- activations

During inference, the major memory consumers are usually:
- parameters
- KV cache
- temporary activations and workspace

So in inference, the part that keeps growing is usually the KV cache, not gradients.

### 2. KV cache grows with context length

In autoregressive generation, each new token stores its Key and Value for reuse by future tokens.

That means:
- longer context -> larger cache
- larger batch -> larger cache
- deeper model -> larger cache
- more heads / larger head dimension -> larger cache

### 3. Long-context inference is not only slower compute

The real issue is often that history becomes expensive to keep around.

This is why different optimizations exist:
- FlashAttention reduces compute and intermediate matrix pressure
- PagedAttention improves how KV cache is organized and allocated
- Quantization reduces parameter and some memory pressure

## A simple estimation rule

For decoder-only models, a useful rough mental model is:

```text
KV Cache size ∝ layers × sequence length × batch size × hidden/head-related dimensions
```

You do not need the full formula on this page. What matters is the intuition:
- doubling sequence length noticeably increases memory
- doubling batch size also increases memory
- deeper models make KV cache harder to ignore

## Common mistakes

- confusing KV cache with training-time gradient storage
- looking only at parameter count and ignoring inference history
- thinking long-context inference is just a compute problem
- assuming "it can generate" means "it can keep generating stably"

## After studying this page, you should be able to answer

- why inference memory grows with generation length
- why long-context inference becomes difficult
- why vLLM emphasizes PagedAttention
- why Chapter 2 studies `Attention`, `FlashAttention`, and `PagedAttention` together

## Connection to later chapters

- **Chapter 2: Attention MHA/GQA**  
  You will see how head count, KV sharing, and cache size are related

- **Chapter 2: FlashAttention**  
  You will see how compute optimization reduces intermediate matrix pressure, but does not automatically solve KV cache organization

- **Chapter 2: vLLM PagedAttention**  
  You will see KV cache management become a real engineering problem

## Summary

This page has one job:
- explain what KV cache is
- explain why it grows
- explain why Chapter 2 needs to deal with it explicitly

If you can already separate "parameter memory" from "inference cache memory", you have the basic intuition needed to enter Chapter 2.

