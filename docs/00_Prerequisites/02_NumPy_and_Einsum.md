# 02. NumPy and Einsum | NumPy 与 Einsum

**难度：** Easy | **环境：** CPU-first | **标签：** `NumPy`, `广播`, `einsum` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/02_NumPy_and_Einsum.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：先看 shape，再看公式；能把简单循环改写成数组运算；能用 `einsum` 表达常见的维度关系。

**关键词：** `ndarray`, `shape`, `einsum`

## 前置阅读
**导语：** 先看 0A 组页，把 Part 0 里 Python 基础和数组思维的边界对齐，再进入这一页会更顺。
- [01. Python Essentials for LLM | Python 基础与数据表示](./01_Python_Essentials_for_LLM.md)
- [0A 组页](./0A.md)
- [01. Data Types and Precision | 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)

## 相关阅读
**导语：** 本页先把 ndarray、广播和 einsum 的最小判断讲清楚；如果想继续把对象封装和配置 / I/O 的写法补完整，可以顺着看下面这些页。
- [03. Python OOP and Utility Patterns | Python 面向对象与工具模式](./03_Python_OOP_and_Utility_Patterns.md)
- [03. GPU Architecture and Memory | GPU 物理架构、内存层级与核心硬件单元](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)

## Q1：ndarray、索引和广播分别解决什么问题？

这一页先把三个最常见的 NumPy 判断立住：数据是不是同质的、能不能按维度切片、能不能用广播省掉手写循环。后面你看到 `mask`、`shape`、`batch` 时，先回到这三个判断。


```python
seq_len = 10
np.triu(np.ones((seq_len, seq_len), dtype=np.int32), k=1)
```


```python
import numpy as np


def build_causal_mask(seq_len):
    """返回 causal mask，shape = [seq_len, seq_len]。"""
    upper = np.triu(np.ones((seq_len, seq_len), dtype=np.int32), k=1)
    return upper == 0


def matmul_einsum(a, b):
    """使用 einsum 实现二维矩阵乘法。"""
    return np.einsum('ij,jk->ik', a, b)


def batch_attention_scores(q, k):
    """计算 batch attention scores，shape = [b, q, k]。"""
    return np.einsum('bqd,bkd->bqk', q, k)


def rms_normalize(x, eps=1e-6):
    """沿最后一维做 RMSNorm 风格归一化。"""
    denom = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    return x / denom

```

## Q1验证：因果掩码和矩阵乘法

这里先验证两个最常见的场景：一个是 mask 形状是否正确，一个是 `einsum` 能不能替代普通矩阵乘法。


```python
def test_build_causal_mask():
    mask = build_causal_mask(4)
    expected = np.array([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ])
    assert np.array_equal(mask, expected)
    print('✅ build_causal_mask 通过')


def test_matmul_einsum():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = matmul_einsum(a, b)
    expected = np.array([[19, 22], [43, 50]])
    assert np.array_equal(result, expected)
    print('✅ matmul_einsum 通过')


test_build_causal_mask()
test_matmul_einsum()

```

## Q2：什么时候该用 `einsum` 直接表达维度关系？

当你已经能看清 `B / T / D / H` 这些维度时，`einsum` 的作用不是炫技，而是把维度关系直接写在式子里，避免先写一堆 reshape 再猜结果对不对。


```python
def test_batch_attention_scores():
    q = np.array([[[1, 0], [0, 1]]])
    k = np.array([[[1, 2], [3, 4]]])
    result = batch_attention_scores(q, k)
    expected = np.array([[[1, 3], [2, 4]]])
    assert np.array_equal(result, expected)
    print('✅ batch_attention_scores 通过')


def test_rms_normalize():
    x = np.array([[3.0, 4.0]])
    y = rms_normalize(x, eps=0.0)
    expected = np.array([[0.84852814, 1.13137085]])
    assert np.allclose(y, expected)
    print('✅ rms_normalize 通过')


test_batch_attention_scores()
test_rms_normalize()

q = np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])
k = np.array([[[1.0, 2.0, 0.0], [3.0, 0.0, 1.0]]])
print('Attention scores shape:', batch_attention_scores(q, k).shape)
print(batch_attention_scores(q, k))
print('Causal mask:')
print(build_causal_mask(5).astype(int))

```

## Q3：什么时候必须先看 shape，再看公式？

后面进 PyTorch、Attention 和更复杂的张量操作时，先看 shape 再看公式通常更稳。只要你能先确认 batch、seq、feature 和 head 这些维度，后面的 reshape、transpose 和广播就不容易写错。


```python
def split_heads(x, num_heads):
    b, t, d = x.shape
    assert d % num_heads == 0
    head_dim = d // num_heads
    return x.reshape(b, t, num_heads, head_dim).transpose(0, 2, 1, 3)


def merge_heads(x):
    b, h, t, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * head_dim)


def add_feature_bias(x, bias):
    return x + bias

```

## Q3验证：shape contract 和 head 拆分是否正确？

这里直接看两件事：`(B, T, D)` 的数据加上 feature bias 后是否还能对齐，以及 split / merge heads 后 shape 是否能回到原样。


```python
x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
bias = np.array([1, 10, 100, 1000])
y = add_feature_bias(x, bias)

print('原始 shape：', x.shape)
print('加 bias 后的第一行：', y[0, 0])
assert np.array_equal(y[0, 0], x[0, 0] + bias)

heads = split_heads(x, 2)
print('split 后 shape：', heads.shape)
merged = merge_heads(heads)
print('merge 后 shape：', merged.shape)
assert np.array_equal(merged, x)
print('✅ shape contract 通过')

```

## 延伸：`einsum` 是个很通用的算子

前面用 `einsum` 写了矩阵乘和 attention scores。其实同一套「下标记法」还能表达一大类常见运算，规则只有三条：

1. **箭头左边**是输入下标，**右边**是输出下标。
2. **在输入里出现、但输出里消失的字母** → 沿这个维度做乘加求和（reduce）。
3. **输出里保留的字母** → 逐元素对齐，不求和。

只要把维度关系写进式子，就不用先 reshape / transpose 再猜结果对不对。下面一次性演示几个「一行搞定」的通用算子：

| 式子 | 含义 | 等价写法 |
|---|---|---|
| `'ij->ji'` | 转置（transpose，交换下标，不求和） | `A.T` |
| `'ii->'` | 求迹（trace，重复下标走对角线 → 标量） | `np.trace(A)` |
| `'i,j->ij'` | 外积（outer，两下标都保留 → 每对相乘） | `np.outer(u, v)` |
| `'ij->i'` | 按维求和（j 消失 → 每行求和） | `A.sum(axis=1)` |
| `'ij->'` | 全部求和（i、j 都消失） | `A.sum()` |
| `'i,i->'` | 内积（共有下标 i 消失即乘加） | `u @ v` |


```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])          # shape (2, 3)

# 1) 转置 transpose：'ij->ji'，交换下标顺序，不做求和
t = np.einsum('ij->ji', A)
assert np.array_equal(t, A.T)

# 2) 求迹 trace：'ii->j'，重复下标 i 表示走对角线，输出为空 -> 标量
S = np.array([[1, 2],
              [3, 4]])
tr = np.einsum('ii->', S)
print("tr shape", tr.shape)
assert tr == np.trace(S) == 5

# 3) 外积 outer product：'i,j->ij'，两个下标都保留 -> 每对元素相乘
u = np.array([1, 2, 3])
v = np.array([10, 20])
outer = np.einsum('i,j->ij', u, v)
assert np.array_equal(outer, np.outer(u, v))

# 4) 按维求和 sum along axis：'ij->i'，下标 j 消失 -> 沿 j 求和（每行求和）
row_sum = np.einsum('ij->i', A)
assert np.array_equal(row_sum, A.sum(axis=1))    # [6, 15]

# 5) 全部求和：'ij->'，i、j 都消失 -> 所有元素相加
total = np.einsum('ij->', A)
assert total == A.sum() == 21

# 6) 内积 dot：'i,i->'，共有下标 i 消失即做乘加
dot = np.einsum('i,i->', u, np.array([1, 1, 1]))
assert dot == u.sum() == 6

print("transpose  ij->ji :", t.tolist())
print("trace      ii->   :", tr)
print("outer      i,j->ij:", outer.tolist())
print("row sum    ij->i  :", row_sum.tolist())
print("total sum  ij->   :", total)
print("dot        i,i->  :", dot)
print('✅ einsum 通用算子演示通过')

```

### 底层判据：einsum 就是一段 for 循环

上面的算子表看着零散，其实背后只有**一条规则**。einsum 的本质是：**每个不同的字母 = 一层 for 循环**，整个运算就是这段伪代码：

```
# 遍历「所有出现过的字母」的每一种取值组合（输出字母 + 被求和的字母 都算）
for i, j, k, ...:                       # 每个字母一层循环
    prod = 1
    for 每个输入张量 operand:
        prod *= operand[它自己的那几个字母]   # 相同字母 → 对齐到同一个下标值
    output[输出字母] += prod             # 累加到输出对应位置
```

从这段循环能读出全部判据：

1. **相同字母对齐相乘** —— 不同输入里同名字母取同一个下标值（`ij,jk` 里两个 `j` 必须是同一个 j）。
2. **输出里有的字母 = 自由维度**，保留，不求和。
3. **输出里没有的字母 = 求和维度**，`+=` 把它累加掉。

> 「要不要 reduce」的**唯一判据** = **这个字母在不在箭头右边**。不在 → 沿它求和；在 → 保留。注意是「不在输出」，而不是「被两边共享」。

### 关键澄清：`+=` 一直都在，区别是「撞不撞车」

容易被 `+=` 绕晕：**两个例子都用 `output[...] += prod`，这行永远执行**。真正决定「有没有累加」的，是——

> **对一个固定的输出格子，有几种 `(i,j,k,...)` 组合会落到它身上？**

- 落上去的组合**只有 1 种** → `+=` 只跑一次 → 等价于普通赋值 `=`，没有真正累加。
- 落上去的组合**有多种** → `+=` 跑多次撞进同一格 → 真的把多项加起来。

而「几种组合落到同一格」完全由**不在输出里的字母（缺席字母）**决定：输出里的字母被格子坐标钉死了，缺席字母可以自由取值，**它的每个取值都砸向同一个格子**。

**每个输出格子被写的次数 = 缺席字母各自取值个数的乘积：**

| 情况 | 缺席字母 | 每格被 `+=` 写几次 | 效果 |
|---|---|---|---|
| `'ij,jk->ijk'` | 无（i、j、k 全在输出） | 1 次 | = 赋值，无累加 |
| `'ij,jk->ik'` | `j`（取 0、1 两个值） | 2 次 | 沿 `j` 累加 = 矩阵乘 |

以 `output[0,0]` 为例看 `ik`：`i=0、k=0` 被钉死，`j` 自由取 0、1，两次循环都砸到同一格 →
`output[0,0] += A[0,0]*B[0,0]`（j=0），再 `+= A[0,1]*B[1,0]`（j=1）= `5 + 14 = 19`。

一句话:**`+=` 永远在;输出里缺席的字母越多,同一格被砸得越多,求和范围越大;一个都不缺,就退化成逐格赋值。**

### 对照：留住 `j` vs 沿 `j` 求和

用同一组 `A`、`B`，只改输出写不写 `j`：

- `'ij,jk->ijk'`：`j` **在**输出 → 不求和，每个乘积项 `A[i,j]*B[j,k]` 摊开保留（shape `(i,j,k)`）。
- `'ij,jk->ik'`：`j` **不在**输出 → 沿 `j` 累加，正好就是矩阵乘（shape `(i,k)`）。

两者关系:**`ik` = `ijk` 沿 `j`(axis=1) 求和**。留 `j` 停在摊开状态，丢 `j` 就多做一步累加。下面第一段 demo 直接对照两者;第二段手写「for 循环 + 数每格被写几次」,把 `+=` 撞车的过程摊开看。


```python
A = np.array([[1, 2],
              [3, 4]])          # shape (i, j)
B = np.array([[5, 6],
              [7, 8]])          # shape (j, k)

# 例1：'ij,jk->ijk' —— j 在输出，不求和，每个乘积项 A[i,j]*B[j,k] 摊开保留
ijk = np.einsum('ij,jk->ijk', A, B)     # shape (2, 2, 2)

# 例2：'ij,jk->ik'  —— j 不在输出，沿 j 累加，正好就是矩阵乘
ik = np.einsum('ij,jk->ik', A, B)       # shape (2, 2)

print('ijk (保留 j)：')
print(ijk)
print('ik  (沿 j 求和)：')
print(ik)

# 关键关系：例2 = 例1 沿 j(axis=1) 求和；丢掉 j 只是多做一步累加
assert np.array_equal(ijk.sum(axis=1), ik)
assert np.array_equal(ik, A @ B)
print('✅ ijk.sum(axis=1) == ik == A @ B')

```


```python
# 手写「一段 for 循环」版 einsum：+= 永远执行，只统计每个输出格子被写了几次
def naive_einsum(out_letters):
    """out_letters: 输出保留哪些字母，例如 'ijk' 或 'ik'。字母 i,j,k 各取 {0,1}。"""
    shape = tuple(2 for _ in out_letters)
    out = np.zeros(shape, dtype=int)
    hits = np.zeros(shape, dtype=int)          # 每个格子被 += 命中的次数
    for i in range(2):
        for j in range(2):
            for k in range(2):                 # 每个字母一层循环
                prod = A[i, j] * B[j, k]        # 相同字母 j 对齐相乘
                idx = tuple({'i': i, 'j': j, 'k': k}[c] for c in out_letters)
                out[idx] += prod                # += 永远执行
                hits[idx] += 1

    missing = [c for c in 'ijk' if c not in out_letters] or ['无']
    print(f"输出 '{out_letters}'：缺席字母={missing}，每格被 += 写的次数=")
    print(hits)
    assert np.array_equal(out, np.einsum(f'ij,jk->{out_letters}', A, B))
    return out


naive_einsum('ijk')     # 无缺席 → 每格写 1 次 → 退化成逐格赋值
print()
naive_einsum('ik')      # j 缺席(2 个取值) → 每格写 2 次 → 沿 j 累加 = 矩阵乘
print('✅ 手写循环与 np.einsum 结果一致；hits 次数 == 缺席字母取值个数之积')

```

### 重复字母 `'ii->'`：同一个字母 = 共用一层循环 → 走对角线

前面 `ij,jk` 里同名的 `j` 是「跨两个输入对齐到同一下标」。**同一个输入里字母重复,是同一条规则用在一个张量的两个槽位上**:两个轴被同一个循环变量索引 → 只走对角线。

普通 `'ij'`——`i`、`j` 两个独立循环,遍历整个网格：

```
for i in range(n):
    for j in range(m):
        ... A[i, j]          # 全部 n×m 个元素
```

`'ii'`——只有一个字母 `i`,**只有一层循环**,同一个 `i` 塞进两个轴：

```
for i in range(n):           # 一个字母 → 一层循环
    total += A[i, i]         # 同一个 i 索引两个轴 → 只走对角线 A[0,0],A[1,1],...
```

之后照旧看输出缺不缺席：

- `'ii->'`：`i` 不在输出 → 对角线再求和 → **迹 trace**。
- `'ii->i'`：同样走对角线,`i` 在输出 → 不求和,落到不同格 → **对角线向量 `diag`**。


```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 'ii->'：重复字母 i 只有一层循环，同一个 i 同时索引两个轴 → 走对角线，再求和
total = 0
for i in range(3):
    total += M[i, i]          # M[0,0]+M[1,1]+M[2,2] = 1+5+9
print("'ii->'  手写:", total, "| np.einsum:", np.einsum('ii->', M), "| np.trace:", np.trace(M))
assert total == np.einsum('ii->', M) == np.trace(M) == 15

# 'ii->i'：同样走对角线，但 i 在输出 → 不求和，每个 i 落到不同输出格
diag = np.zeros(3, dtype=int)
for i in range(3):
    diag[i] += M[i, i]        # 落到 diag[i]，互不相撞 → 无累加
print("'ii->i' 手写:", diag.tolist(), "| np.einsum:", np.einsum('ii->i', M).tolist())
assert np.array_equal(diag, np.einsum('ii->i', M))
print('✅ 重复字母 = 共用一层循环走对角线；ii-> 再沿 i 求和，ii->i 保留')

```

## 推广：同一套「字母进出」记账法 → einops 的 rearrange / reduce / repeat

`einsum` 的记账法不止管 `einsum`。einops 全家也是**给每个轴起字母名,把左边(输入)和右边(输出)的字母一比,谁进谁出决定动作**。字母进出恰好三种情形,对应 einops 三个操作:

| 字母出现在 | 动作 | 元素映射 | einops | einsum 对应 |
|---|---|---|---|---|
| **左右都有** | 保留 / 重排 / 拆合 | **双射**（1↔1） | `rearrange` | 输出保留的字母 |
| **只在左**（输出缺席） | 沿它聚合 | **多对一** | `reduce` | 缺席字母 `+=` 累加 |
| **只在右**（输入没有，新轴） | 广播复制 | **一对多** | `repeat` | einsum 造不出新轴 |

- `rearrange`：左右字母集合相同，只换顺序/拆合 → **每个输出格恰好来自 1 个输入格**，元素数不变（对应 einsum 全字母都在输出、每格 `+=` 只 1 次的情形）。
- `reduce`：某字母只在左边 → 沿它聚合 → **多个输入格砸进同一输出格**。这正是 einsum 缺席字母 `+=` 累加的同一条规则，只是聚合可选 `sum/mean/max`。**所以 `einsum` ≈「先按共有字母相乘，再 reduce」**。
- `repeat`：某字母只在右边（输入没有的新轴）→ **一个输入格复制到多个输出格**，是 `reduce` 的镜像反向。einsum 做不到——它的输出字母必须在某个输入里出现过。

> `+=` 的「撞车累加」只字面适用于 `reduce`；`rearrange` 永远不撞车（双射），`repeat` 是反着的扇出（一个源被读多次）。


```python
from einops import rearrange, reduce, repeat

x = np.arange(6).reshape(2, 3)   # 'h w', h=2, w=3
print('x =\n', x, ' shape(h,w)=', x.shape, ' 元素数:', x.size)

# 1) rearrange：左右字母集合相同（只换位）→ 双射，每个输出格来自 1 个输入格，size 不变
r = rearrange(x, 'h w -> w h')                 # 转置
assert r.size == x.size and np.array_equal(r, x.T)
print("\nrearrange 'h w -> w h'：字母无进出 → 双射，size", x.size, '→', r.size)

# 2) reduce：h 只在左边（输出缺席）→ 沿 h 聚合 → 多对一（= einsum 缺席字母 += 累加）
red = reduce(x, 'h w -> w', 'sum')             # 沿 h 求和
assert np.array_equal(red, x.sum(axis=0))
print("reduce   'h w -> w' sum：h 缺席 → 每个输出格吃进 h 的", x.shape[0], '个输入格（多对一）:', red.tolist())

# 3) repeat：c 只在右边（新轴）→ 广播复制 → 一对多（reduce 的反向）
rep = repeat(x, 'h w -> h w c', c=2)           # 复制 2 份
assert rep.size == x.size * 2 and np.array_equal(rep[:, :, 0], rep[:, :, 1])
print("repeat   'h w -> h w c', c=2：c 新增 → 每个输入格复制到 2 个输出格（一对多），size", x.size, '→', rep.size)
print('\n✅ rearrange 双射 / reduce 多对一 / repeat 一对多，全部与 numpy 等价写法一致')

```
