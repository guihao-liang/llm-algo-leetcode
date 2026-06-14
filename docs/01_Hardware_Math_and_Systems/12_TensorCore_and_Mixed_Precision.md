# 12. Tensor Core and Mixed Precision

**Difficulty:** Medium | **Tags:** `Tensor Core`, `Mixed Precision`, `Throughput` | **Audience:** Learners preparing to enter Chapter 2

This page connects the data-format, memory, and parameter-count intuition from Chapter 1 to the practical questions of "why mixed precision is faster" and "why quantization is worth it". Many beginners know the names FP32, FP16, BF16, and INT8, but they do not yet know why those formats change throughput. Tensor Core and mixed precision are the bridge that explains that relationship.

## Why this is a bridge page

- Chapter 1 already covered the basic footprint of FP32 / FP16 / BF16 / INT8
- Chapter 2 will move into quantization, training, inference cost, and memory optimization
- If readers do not understand why mixed precision improves throughput and reduces memory pressure, later optimizations feel like "just changing formats"
- So this page exists to explain the relationship between precision and speed

## The intuition you should build first

### 1. Higher precision is not always better

Higher precision usually means:
- larger memory usage
- more storage and transfer cost
- not necessarily better practical value

So in training and inference, people choose formats based on the task instead of always preferring FP32.

### 2. Tensor Core represents a hardware acceleration path

You do not need all the hardware details here, but you should know:
- GPUs have dedicated acceleration paths for matrix multiplication
- mixed precision often aligns better with those high-throughput paths
- so it is not only about saving memory, but also about running faster

### 3. Mixed precision is about balancing precision and throughput

In practice, the common idea is:
- use lower precision for most of the heavy computation
- keep higher precision where it matters
- balance overall efficiency and quality

That is why mixed precision is not "cutting corners"; it is a systems-level tradeoff.

## A common understanding path

You can think about it like this:

```text
higher precision -> higher storage/transfer cost -> not always faster
lower precision + carefully preserved critical state -> better throughput and resource usage
```

This intuition is more important than memorizing the exact bit widths of every dtype.  
Chapter 2's quantization, training cost, and inference optimization all rely on this idea.

## Common mistakes

- thinking FP32 is always best
- thinking mixed precision is only for saving memory
- treating Tensor Core as just a name, without connecting it to throughput
- assuming quantization is only compression, without considering hardware path, throughput, and error control

## After studying this page, you should be able to answer

- why training and inference do not always use the highest precision
- why mixed precision can improve overall efficiency
- why certain formats fit certain scenarios better
- why Chapter 2's quantization and training-cost chapters come after Chapter 1

## Connection to later chapters

- **Chapter 2: Quantization W8A16 / QLoRA / 4-bit Quantization**  
  You will see how low-precision representations directly affect memory and compute efficiency

- **Chapter 2: Training Loop / SFT / LoRA / RLHF**  
  You will see how mixed precision affects training stability and resource usage

- **Chapter 2: Inference Optimization**  
  You will see how precision choices affect inference throughput and memory pressure

## Summary

This page has one job:
- explain the relationship between precision and throughput
- explain why mixed precision matters
- explain why Chapter 2 will use these ideas repeatedly

If you already know that "higher precision is not always better" and that you must balance precision and throughput, you already have the key intuition needed to enter Chapter 2.

