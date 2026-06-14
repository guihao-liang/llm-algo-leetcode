# 讨论题 23：Tensor Core 深度剖析 (Tensor Core Deep Dive)

**难度：** Hard | **标签：** `GPU`, `矩阵乘法`, `混合精度` | **目标人群：** 想把“Tensor Core 很快”理解到能做性能判断的学习者

这一页不打算把 GPU ISA 讲到汇编级别，而是想把 Tensor Core 的核心问题说清楚：
- 它到底加速了什么
- 为什么对 GEMM 特别有效
- 为什么输入形状、数据类型和 tile 切分会影响利用率

前置阅读建议先看 `1B-01 单卡硬件与访存优化`，先把 SM、Warp、Shared Memory 和 HBM 的基本层级关系搞清楚，再看这页会更顺。

## Q1：Tensor Core 本质上是什么？

<details>
<summary>点击展开查看解析</summary>

Tensor Core 可以理解成专门为矩阵乘加设计的硬件单元。

普通 CUDA Core 更像标量计算单元，而 Tensor Core 更像“矩阵乘法加速器”。它的核心价值不是“更快做一个标量运算”，而是用更大的粒度一次完成更多矩阵乘加。

这意味着：
- 当任务是 GEMM 或 Attention 中的大量矩阵乘法时，Tensor Core 很有优势
- 当任务是碎片化、分支多、复用低的算子时，Tensor Core 的优势会下降

一个数量级直觉表可以帮助你判断它到底快在哪里：

| 架构 | CUDA Core FP32 | Tensor Core FP16 | Tensor Core FP8 | 直觉 |
| --- | --- | --- | --- | --- |
| V100 | 15.7 TFLOPs | 125 TFLOPs | - | 引入 Tensor Core 后，训练开始明显受益 |
| A100 | 19.5 TFLOPs | 312 TFLOPs | - | Tensor Core 已经成为主力算力来源 |
| H100 | 67 TFLOPs | 989 TFLOPs | 1979 TFLOPs | FP8 进一步把吞吐拉高 |

如果再看代际演进，Tensor Core 的功能也在变：

| 架构 | Tensor Core 代际 | 新增精度 / 特性 | 说明 |
| --- | --- | --- | --- |
| Volta (V100) | 1st Gen | FP16 | 首次引入 Tensor Core |
| Turing (T4 / RTX) | 2nd Gen | INT8 / INT4 | 推理侧更受益 |
| Ampere (A100) | 3rd Gen | BF16 / TF32 | 稀疏性支持增强 |
| Hopper (H100) | 4th Gen | FP8 | 引入 Transformer Engine、WGMMA、TMA |
| Blackwell (B100) | 5th Gen | FP4 | 更低位宽继续下探 |

所以 Tensor Core 快，不是因为它“神秘”，而是因为它匹配了深度学习里最常见的计算形态。
</details>

## Q2：为什么精度和吞吐可以同时受益？

<details>
<summary>点击展开查看解析</summary>

Tensor Core 常见的工作方式是：
- 输入使用较低精度，如 FP16 / BF16 / TF32 / FP8
- 累加使用更高精度

这样做的原因是：
- 低精度输入能减少搬运压力、提高吞吐
- 高精度累加能保住结果稳定性

Hopper 之后还有两个值得记住的新特性：
- **WGMMA**：Warpgroup MMA，允许更多线程协同完成矩阵乘法，减少同步开销
- **TMA**：Tensor Memory Accelerator，负责更高效的异步数据搬运，让计算和传输更容易重叠

这也是为什么 Tensor Core 经常和混合精度训练一起出现。它不是简单把精度压低，而是把“足够低的输入精度”和“足够稳的累加精度”组合起来。
</details>

## Q3：为什么 Tensor Core 利用率不是随便就能跑满？

<details>
<summary>点击展开查看解析</summary>

Tensor Core 的高吞吐通常依赖几个条件：
- 矩阵形状要适合 tile 切分
- 数据布局要尽量规整
- 输入维度要够大，才能摊薄调度和搬运开销
- 上层 kernel 要避免频繁的中间写回

更具体一点，Tensor Core 友好的矩阵通常需要：
- `M / N / K` 至少达到较规整的 tile 尺寸，太小会回退到 CUDA Core
- 常见 tile 尺寸包括 `16×16×16`、`32×8×16`、`64×32×16`
- 从经验上看，很多实现会要求维度至少是 `8` 或 `16` 的倍数，才能更稳定地触发高效 MMA 路径

如果矩阵太小、形状太碎，或者数据布局不适合，Tensor Core 就可能“有硬件，但吃不满”。

这也是为什么性能优化常常不是“换一个更快的算子名”这么简单，而是要把形状、layout、fusion 和 memory access 一起考虑。
</details>

## Q4：这页最该避免的误区是什么？

<details>
<summary>点击展开查看解析</summary>

- **“Tensor Core 就是更快的 CUDA Core”**  
  不对。它是不同粒度的硬件设计。

- **“只要用上 Tensor Core 就一定快”**  
  不对。输入形状、内存布局和 kernel 组织方式都会影响利用率。

- **“Tensor Core 只和训练有关”**  
  不对。推理中的 GEMM、投影层、Attention 相关矩阵乘法也会受益。

- **“低精度一定差”**  
  不对。对很多深度学习任务，合理的低精度配合更高效的累加反而更实用。
</details>

## 小结

如果你能回答下面两句话，这页就算抓住了：

- Tensor Core 是专门加速矩阵乘加的硬件，不是普通标量 ALU
- Tensor Core 利用率取决于矩阵形状、数据布局和算子组织方式

## 配合练习

这一页建议和后面的 `1B-03` 练习一起做，练习目标可以这样设计：

- 用 Triton 测 Tensor Core 利用率，比较不同矩阵形状下的吞吐变化
- 对比 FP16 和 FP8 的利用率差异，观察吞吐和精度的权衡

如果后面要补 Notebook，建议先围绕“矩阵形状是否触发 Tensor Core”和“不同精度下的吞吐差异”两个方向展开。
