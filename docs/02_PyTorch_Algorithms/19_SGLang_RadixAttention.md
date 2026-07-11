# 19. SGLang RadixAttention | SGLang 与 RadixAttention: 突破 vLLM 多轮对话瓶颈

**难度：** Medium | **标签：** `SGLang`, `Radix Tree`, `KV Cache` | **目标人群：** 模型部署与推理引擎开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/19_SGLang_RadixAttention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


上一节我们学习了业界广泛使用的 **vLLM (PagedAttention)**。它准确解决了单次生成长文本时的**内存碎片**问题。
但在实际的生产环境中，我们经常遇到以下场景：
- **庞大且共享的 System Prompt**（如几百字的设定语，每个用户请求都带着）。
- **多轮对话（Multi-turn Chat）**（只增加了最后一句用户提问，前面几万字的聊天记录都是相同的）。
- **Few-shot Prompting**（给所有请求都塞入相同的 few-shot 示例）。

vLLM 会对每一个请求**从头开始（重新计算）**这些共享前缀的 KV Cache，这浪费了大量的时间（导致 TTFT 首字响应极慢）和显存。

**SGLang (LMSYS, 2024)** 的提出，它的核心创新 **RadixAttention** 引入了**基数树 (Radix Tree)** 来管理系统的 KV Cache。当系统发现新的请求和旧请求有着相同的前缀（Prefix）时，它会**直接复用**树节点里的 KV Cache，完全跳过了重复的 Prefill 阶段！
### Step 1: 核心机制对比

> **vLLM PagedAttention：线性的分页存储**
> vLLM 就像操作系统的虚拟内存。每个请求有一个页表，它能很好地解决碎片，但请求与请求之间的页表是隔离的。就算两个请求的 Prompt 一模一样，它们也会各占一份显存，各算一次。

> **SGLang RadixAttention：基于基数树的共享路由**
> 系统维护了一棵全局的树。树的每一条边代表一段 Token 序列，节点里存着这段序列对应的 KV Cache 物理块指针。
> 当新请求到来时，SGLang 会用它的 Prompt 去这棵树里做**最长前缀匹配 (Longest Prefix Match)**。
> 匹配到的部分直接拿来用，没匹配到的部分再去计算并作为新分支挂在树上。
### Step 2: 动手实战 —— 模拟 Radix Tree 前缀匹配

为了让你深刻理解 SGLang 的调度思想，我们将用 Python 原生数据结构，亲手模拟一个非常简化的 Radix Tree 路由管理器。

**要求**：完成 `match_prefix` 函数，在全局 KV 树中寻找当前请求的最长前缀，返回可以省去的重计算长度（Hit Length）。

```python
import torch
```


```python
class TreeNode:
    def __init__(self, key_tokens):
        self.key_tokens = key_tokens  # 这条边上的 Token 序列 (如 [101, 532, 789])
        self.children = []            # 子节点列表
        self.kv_cache_ptr = None      # 模拟指向物理 KV Cache 的指针

class SimpleRadixCache:
    def __init__(self):
        # 根节点是空的
        self.root = TreeNode([])
        
    def insert(self, tokens):
        """简单模拟向基数树中插入完整的请求。"""
        # 简化的插入逻辑（不涉及分裂裂变，仅为演示层级添加）
        # 假设当前系统只有一个 System Prompt，我们将其挂在根节点下
        node = TreeNode(tokens)
        self.root.children.append(node)
        
    def match_prefix(self, prompt_tokens):
        """
        在现有树中，为新的 prompt_tokens 寻找最长的匹配前缀。
        如果前 N 个 token 完全一致，说明这 N 个 token 的 KV Cache 可以直接复用！
        """
        # 为了教学，我们只做单层子节点的暴力匹配
        best_match_len = 0
        
        # ==========================================
        # TODO: 遍历 self.root.children，寻找能与 prompt_tokens 匹配的最长前缀
        # 如果子节点的 token 序列是 [A, B, C]，而 prompt 是 [A, B, C, D, E]
        # 那么最佳匹配长度就是 3。
        # ==========================================
        # YOUR CODE HERE
        
        return best_match_len
```


```python
# 测试你的实现
def test_radix_attention():
    try:
        cache = SimpleRadixCache()
        
        # 1. 模拟系统初始化：所有的聊天都带有一段长达 100 词的“系统人设”
        system_prompt = list(range(100)) # 用 0~99 的数字模拟 Token
        cache.insert(system_prompt)
        
        # 2. 用户 A 来了：带有系统人设，并问了一句话
        user_a_prompt = list(range(100)) + [1001, 1002, 1003]
        
        match_len_a = cache.match_prefix(user_a_prompt)
        assert match_len_a == 100, "匹配失败！系统人设的 100 个 Token 应该完全被复用！"
        print(f"✅ 用户 A 命中前缀缓存！原本需要计算 {len(user_a_prompt)} 个 token，现在只需计算最后 {len(user_a_prompt) - match_len_a} 个！")
        
        # 3. 用户 B 来了：一个完全不同的、没有系统人设的请求
        user_b_prompt = [9999, 8888, 7777]
        match_len_b = cache.match_prefix(user_b_prompt)
        assert match_len_b == 0, "错误匹配！不该匹配到任何东西。"
        print("✅ 用户 B 正常 fallback，未命中缓存。")
        
        print("\n 所有测试通过！这正是 SGLang 让大模型推理首字响应飞升 10 倍的底层秘密！")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")

test_radix_attention()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---
## 参考代码与解析

### 代码

```python
class TreeNode:
    def __init__(self, key_tokens):
        self.key_tokens = key_tokens
        self.children = []
        self.kv_cache_ptr = None

class SimpleRadixCache:
    def __init__(self):
        self.root = TreeNode([])
        
    def insert(self, tokens):
        """简单模拟向基数树中插入完整的请求。"""
        node = TreeNode(tokens)
        self.root.children.append(node)
        
    def match_prefix(self, prompt_tokens):
        """
        在现有树中，为新的 prompt_tokens 寻找最长的匹配前缀。
        """
        best_match_len = 0
        
        # TODO: 遍历 self.root.children，寻找能与 prompt_tokens 匹配的最长前缀
        for child in self.root.children:
            cached_tokens = child.key_tokens
            match_len = 0
            
            # 逐个对比 Token，计算最长连续相同前缀
            while match_len < len(cached_tokens) and match_len < len(prompt_tokens):
                if cached_tokens[match_len] == prompt_tokens[match_len]:
                    match_len += 1
                else:
                    break
            
            if match_len > best_match_len:
                best_match_len = match_len
                
        return best_match_len
```

### 解析

**1. TODO: 最长前缀匹配算法**
- **实现方式**：遍历根节点的所有子节点，逐个 token 比较，找到最长的连续匹配前缀
- **关键点**：使用 while 循环逐位比较，遇到不匹配立即停止
- **技术细节**：需要同时检查两个边界条件：`match_len < len(cached_tokens)` 和 `match_len < len(prompt_tokens)`

**核心算法流程**
1. 初始化 `best_match_len = 0`
2. 遍历树中所有已缓存的节点（`self.root.children`）
3. 对每个节点，从第 0 个 token 开始逐个比较
4. 如果 `cached_tokens[i] == prompt_tokens[i]`，继续比较下一个
5. 如果不匹配或到达边界，停止比较
6. 更新 `best_match_len` 为所有节点中的最大匹配长度

**工程优化要点**
- **TTFT 优化**：通过前缀复用，首字响应时间（Time To First Token）可降低 5-10 倍
- **显存节省**：共享的 System Prompt 只需存储一次，多个请求共享同一份 KV Cache
- **多轮对话加速**：对话历史作为公共前缀，只需计算最新的用户输入
- **树结构优化**：真实的 SGLang 使用更复杂的 Radix Tree，支持节点分裂和合并
- **LRU 淘汰**：当显存不足时，使用 LRU 策略淘汰最久未使用的树节点
- **工业实践**：SGLang 在多轮对话场景下，吞吐量比 vLLM 提升 3-5 倍
---

## 进阶实战：手写一棵「真正的」Radix Tree

前面的 `SimpleRadixCache` 为了教学做了极度简化——它只有**一层子节点**、**暴力遍历**、而且 `insert` 从不分裂。这其实是个「前缀列表」，不是基数树。我们分**两个阶段**由浅入深地把它升级成 SGLang 真实 `RadixCache` 的核心：

- **🔥 Stage 1（本节）：多层树 + 节点分裂**——让树能长很多层、能在一条边的中间停下匹配，并在插入时**按需分裂边**。这是基数树的灵魂。
- **🧊 Stage 2（下一节，写完再加）：LRU 淘汰 + 引用计数**——显存满时淘汰最久未用的叶子，同时锁住正在被请求使用的节点。

---

### Stage 1：多层匹配 + 节点分裂

**为什么需要分裂？** 假设树里已有一条边 `[1,2,3,4,5]`。现在来了 `[1,2,3,8,9]`，它和这条边只共享前缀 `[1,2,3]`。基数树的做法是把那条边**从第 3 个 token 处切开**：

```
分裂前:                     分裂后:
root                        root
 └─ [1,2,3,4,5]              └─ [1,2,3]          <- 新的中间节点(公共前缀)
                                 ├─ [4,5]        <- 旧边的后半段
                                 └─ [8,9]        <- 新请求的分叉后缀
```

这样 `[1,2,3]` 的 KV Cache 就能被两个请求共享了。

**你要实现的接口（本阶段无 LRU）：**

```
RadixCache
 ├─ match_prefix(tokens)      -> (matched_len, last_node)   # 多层 + 边内 最长前缀匹配
 ├─ _split(parent, child, k)  -> mid                        # 在第 k 个 token 处切开一条边
 └─ insert(tokens, value)     -> node                       # 复用前缀，按需分裂，挂新后缀
```

**数据结构约定**（radix 风格）：每个节点的 `children` 是 `dict`，用**这条子边的首 token** 作 key，`O(1)` 找到候选边；`node.key` 是从父节点到本节点这条边上的 token 段；`root.key = []`。辅助函数 `shared_prefix_len(a, b)` 已给你。

> 🎯 下面 cell 里有 `# TODO`，先自己写，卡住了再看参考答案。Stage 2 的 LRU 我们等你把这块跑通了再加。

```python
# Stage 1 练习：多层匹配 + 节点分裂（请补全 TODO）

def shared_prefix_len(a, b):
    """两个 token 列表的最长公共前缀长度（已给你，直接用）。"""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class TreeNode:
    def __init__(self):
        self.children = {}      # dict: 首 token -> 子 TreeNode
        self.parent = None
        self.key = []           # 从 parent 到本节点这条边上的 token 段
        self.value = None       # 模拟 KV Cache 的 slot 索引数组（指向 KV 池的下标）


class RadixCache:
    def __init__(self):
        self.root = TreeNode()
        self.root.key = []

    # ---------- TODO 1: 多层 + 边内 最长前缀匹配 ----------
    def match_prefix(self, tokens):
        """从 root 出发，为 tokens 找最长匹配前缀。
        返回 (matched_len, last_node)：
          - matched_len : 命中的 token 数（可省去的重计算长度）
          - last_node   : 完整走完的最深节点（后面 insert / 锁定会用到）

        思路：
          node = self.root; matched = 0
          循环：用 tokens[matched] 到 node.children 里找候选边 child
                没有 child -> 停
                有 -> 用 shared_prefix_len(child.key, tokens[matched:]) 算这条边命中多少
                     命中整条边 -> matched 前进, 下降 node = child, 继续
                     只命中一部分（边内停）-> matched 加上这部分, break
        """
        # YOUR CODE HERE
        node = self.root
        matched = 0
        for 

    # ---------- TODO 2: 在第 split_at 个 token 处切开一条边 ----------
    def _split(self, parent, child, split_at):
        """把 parent -> child 这条边在 split_at 处切开：
          - 新建中间节点 mid，key = child.key[:split_at]
          - 旧 child 的 key 变成 child.key[split_at:]，挂到 mid 下
          - mid 挂回 parent（首 token 不变，直接替换原来的 child）
        返回 mid。

        提示：
          - 注意维护好 parent / children 指针，用 key[0] 作 children 的 dict key。
          - KV Cache 不用重算、也不搬数据：如果 child.value 非空（是一串指向 KV 池的
            slot 索引），只需把它在 split_at 处切开——mid.value = child.value[:split_at]，
            child.value = child.value[split_at:]。前缀那段 slot 就此被两个分叉共享。
        """
        # YOUR CODE HERE
        raise NotImplementedError

    # ---------- TODO 3: 插入（复用前缀 + 按需分裂）----------
    def insert(self, tokens, value=None):
        """插入 tokens：能复用的前缀就复用，分叉处调用 _split，末端挂新叶子。返回末端节点。

        思路（从 root 逐边往下走，idx 是已消费的 token 数）：
          - 当前 node 没有以 tokens[idx] 开头的子边 -> 直接把 tokens[idx:] 作为新叶子挂上，返回
          - 有子边 child：cpl = shared_prefix_len(child.key, tokens[idx:])
              cpl == len(child.key) -> 整条边复用，idx += cpl，下降 node = child
                                       （若 idx 已到末尾，返回 node）
              cpl <  len(child.key) -> 只共享一部分，先 _split(node, child, cpl) 得到 mid，
                                       idx += cpl，node = mid；
                                       若还有剩余 tokens，把 tokens[idx:] 作为 mid 的新叶子挂上，返回
        """
        # YOUR CODE HERE
        raise NotImplementedError

    # ---------- 调试辅助（已给）----------
    def all_nodes(self):
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children.values())
        return out

    def total_tokens(self):
        return sum(len(n.key) for n in self.all_nodes() if n is not self.root)

    def pretty(self, node=None, depth=0):
        node = node or self.root
        print("  " * depth + ("ROOT" if node is self.root else str(node.key)))
        for c in node.children.values():
            self.pretty(c, depth + 1)

```


```python
# Stage 1 测试：多层匹配 + 节点分裂
def test_stage1_radix_split():
    try:
        # --- 向后兼容：原 notebook 的 system prompt 复用场景 ---
        c = RadixCache()
        c.insert(list(range(100)))
        assert c.match_prefix(list(range(100)) + [1001, 1002, 1003])[0] == 100, \
            "system prompt 100 个 token 应全部命中"
        assert c.match_prefix([9999, 8888, 7777])[0] == 0, "无关请求应 0 命中"
        print("✅ [1] 向后兼容：system prompt 复用")

        # --- 节点分裂：核心考点 ---
        c = RadixCache()
        c.insert([1, 2, 3, 4, 5])
        c.insert([1, 2, 3, 8, 9])   # 与上条共享 [1,2,3]，应在第 3 个 token 处分裂
        node123 = c.root.children[1]
        assert node123.key == [1, 2, 3], f"应分裂出中间节点 [1,2,3]，实际 {node123.key}"
        assert {tuple(ch.key) for ch in node123.children.values()} == {(4, 5), (8, 9)}, \
            "分裂后 [1,2,3] 下应有 [4,5] 和 [8,9] 两个分叉"
        print("✅ [2] 节点分裂：[1,2,3,4,5] + [1,2,3,8,9] -> 公共前缀 [1,2,3] 被抽出")

        # --- 多层 + 边内匹配 ---
        assert c.match_prefix([1, 2, 3, 4, 5])[0] == 5, "整条路径应命中 5"
        assert c.match_prefix([1, 2, 3, 8, 9])[0] == 5, "另一分叉应命中 5"
        assert c.match_prefix([1, 2, 3])[0] == 3, "只到分叉点应命中 3"
        assert c.match_prefix([1, 2, 7])[0] == 2, "边内部分匹配：[1,2,7] 应命中 2"
        assert c.match_prefix([1, 2, 3, 4, 99])[0] == 4, "多层+边内：应命中 4（[1,2,3]+[4]）"
        print("✅ [3] 多层 + 边内匹配：4 种边界情形全部正确")

        # --- 分裂后 total_tokens 不重复计（结构没冗余）---
        assert c.total_tokens() == 3 + 2 + 2, f"树里应恰好 7 个 token，实际 {c.total_tokens()}"
        print("✅ [4] 分裂后无冗余：树中共享前缀只存一份")

        print("\n🎉 Stage 1 全部通过！你已经实现了真正的 radix 分裂。可以进入 Stage 2 (LRU) 了。")
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 发生未知异常: {type(e).__name__}: {e}")

test_stage1_radix_split()

```

---

🛑 **STOP HERE (Stage 1)** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 先自己把 `match_prefix` / `_split` / `insert` 写出来跑通上面的测试。<br>
> 卡住了再往下看 Stage 1 参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

```python
# Stage 1 参考答案：多层匹配 + 节点分裂

def shared_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class TreeNode:
    def __init__(self):
        self.children = {}      # 首 token -> 子 TreeNode
        self.parent = None
        self.key = []           # 从 parent 到本节点这条边上的 token 段
        self.value = None       # 模拟 KV Cache 的 slot 索引数组（每个 token 一个下标，与 key 对齐）


def _slot_slice(value, lo, hi=None):
    """value 是与 token 对齐的 slot 索引数组；None 表示本练习不关心 KV，直接透传。"""
    if value is None:
        return None
    return value[lo:] if hi is None else value[lo:hi]


class RadixCache:
    def __init__(self):
        self.root = TreeNode()
        self.root.key = []

    # ---------- TODO 1: 多层 + 边内 最长前缀匹配 ----------
    def match_prefix(self, tokens):
        node = self.root
        matched = 0
        while matched < len(tokens):
            child = node.children.get(tokens[matched])
            if child is None:
                break                                   # 没有对应子边，停
            cpl = shared_prefix_len(child.key, tokens[matched:])
            matched += cpl
            if cpl == len(child.key):
                node = child                            # 整条边命中，下降继续
            else:
                break                                   # 边内部分命中，前缀止于边中间
        return matched, node

    # ---------- TODO 2: 在第 split_at 个 token 处切开一条边 ----------
    def _split(self, parent, child, split_at):
        mid = TreeNode()
        mid.parent = parent
        mid.key = child.key[:split_at]
        # KV 数据不动、不重算：只把 slot 索引数组在 split_at 处切一刀
        mid.value = _slot_slice(child.value, 0, split_at)   # 前缀的 slot 归 mid（被两分叉共享）
        child.value = _slot_slice(child.value, split_at)    # 后缀的 slot 留给 child

        child.key = child.key[split_at:]                # 旧节点只保留后半段 token
        child.parent = mid
        mid.children[child.key[0]] = child              # child 挂到 mid 下

        parent.children[mid.key[0]] = mid               # 首 token 不变，替换原 child
        return mid

    # ---------- TODO 3: 插入（复用前缀 + 按需分裂）----------
    def insert(self, tokens, value=None):
        node = self.root
        idx = 0
        while idx < len(tokens):
            first = tokens[idx]
            child = node.children.get(first)
            if child is None:                           # 没有子边 -> 剩余后缀挂成新叶子
                leaf = TreeNode()
                leaf.parent = node
                leaf.key = tokens[idx:]
                leaf.value = _slot_slice(value, idx)    # 只取 tokens[idx:] 对应的 slot
                node.children[first] = leaf
                return leaf

            cpl = shared_prefix_len(child.key, tokens[idx:])
            if cpl == len(child.key):                   # 整条边复用，下降（前缀 KV 直接复用）
                idx += cpl
                node = child
                if idx == len(tokens):
                    return node                         # 已完全被现有路径覆盖
            else:                                       # 只共享一部分 -> 分裂
                mid = self._split(node, child, cpl)
                idx += cpl
                node = mid
                if idx == len(tokens):
                    return node
                leaf = TreeNode()                       # 剩余后缀挂到 mid 下
                leaf.parent = mid
                leaf.key = tokens[idx:]
                leaf.value = _slot_slice(value, idx)
                mid.children[tokens[idx]] = leaf
                return leaf
        return node

    # ---------- 调试辅助 ----------
    def all_nodes(self):
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children.values())
        return out

    def total_tokens(self):
        return sum(len(n.key) for n in self.all_nodes() if n is not self.root)

    def pretty(self, node=None, depth=0):
        node = node or self.root
        print("  " * depth + ("ROOT" if node is self.root else str(node.key)))
        for c in node.children.values():
            self.pretty(c, depth + 1)


test_stage1_radix_split()

```

### Stage 1 解析

**1. `match_prefix`：多层 + 边内匹配**
- 从 `root` 出发，每一步用 `tokens[matched]`（当前待匹配的第一个 token）到 `node.children` 里 `O(1)` 找候选边。
- 对候选边 `child`，用 `shared_prefix_len(child.key, tokens[matched:])` 算这条边命中多少个 token：
  - **命中整条边**（`cpl == len(child.key)`）→ 下降到 `child`，继续往深处匹配。
  - **只命中一部分**（边内停）→ 前缀就止于这条边中间，`break`。
- 返回 `(matched_len, last_node)`：`matched_len` 是能复用、可省去重算的 token 数；`last_node` 是完整走完的最深节点。

**2. `_split`：基数树的灵魂**
- 新建中间节点 `mid` 持有公共前缀 `child.key[:k]`；旧 `child` 只保留 `child.key[k:]`。
- 指针重接：`child` 挂到 `mid` 下，`mid` 挂回 `parent`。因为 `mid.key[0] == child` 原来的首 token，所以 `parent.children[mid.key[0]] = mid` 直接**替换**掉原来的 child，不会留下悬挂。
- 复杂度：只动了几个指针，**O(1)**（不含 slice 成本）。

**3. `insert`：复用 + 按需分裂**
- 从 root 逐边下降，`idx` 记已消费 token 数。三种情况：无子边→整段挂新叶子；整条边复用→下降；部分共享→`_split` 后把剩余后缀挂到 `mid` 下。

**❓ 分裂时 KV Cache 怎么办？（高频追问）**
- 关键认知：**KV Cache 的向量数据一个字节都不动，也不重算**。`node.value` 存的不是 KV 向量本身，而是一串**指向 KV 池的 slot 索引**（对应 PagedAttention 那个 `[num_tokens, head_dim]` 池的下标）。
- 分裂 `[1,2,3,4,5]` 这条边，只是把它的索引数组 `value` 在第 k 处**切一刀**：

  ```python
  mid.value   = child.value[:k]   # 前缀 [1,2,3] 的 slot 索引
  child.value = child.value[k:]   # 后缀 [4,5]  的 slot 索引
  ```
- 物理上那些 KV 向量还躺在池子原地，`[1,2,3]` 的 slot 现在被 `mid` 记着、被两个分叉**共享**；`[4,5]` 的 slot 归 `child`。所以分裂是纯**元数据操作**——这正是 RadixAttention 能零成本复用前缀的原因。（本练习里 `value` 用 `None` 占位，参考答案已按「有值就切片」的方式正确处理。）

**下一步（Stage 2）**：给节点加 `ref_count`（正在解码的请求锁住路径）和 `last_access_time`，实现 `evict(num_tokens)` 按 LRU 淘汰未锁叶子——就凑齐了 SGLang `RadixCache` 的完整骨架。
---

## 延伸：对照真实 SGLang 源码（mini-sglang）

上面的练习是 token 级、简化版。真实 SGLang 的 KV 管理是**三层结构**（下面每条都对应 `mini-sglang` 源码，已逐一核对）：

| 组件 | 源码 | 作用 |
|---|---|---|
| **KV 物理池** `_kv_buffer` | `kvcache/mha_pool.py:28` `[2, layers, num_pages, page_size, heads, head_dim]` | 开机一次性预分配，永不 grow/free；`store_cache` 把它 view 成 `[num_pages*page_size, heads*head_dim]` 按 **flat slot 行号** 读写（`kernel/store.py:37`） |
| **page_table** `[max_reqs, max_len]` | `scheduler/cache.py`、`scheduler/table.py` | **每请求**的 token→slot 映射（= vLLM block table） |
| **radix 树** `RadixPrefixCache` | `kvcache/radix_cache.py` | **全局共享**前缀缓存，`node.value` 存 per-token slot 索引 |

### 1. `page_table[req, pos]` 怎么找 slot？`pos` 是什么？

- **`req`（=`table_idx`）**：这个请求在「运行中请求表」里的行号（`TableManager.allocate()` 分配，`table.py:17`）。
- **`pos`**：**token 在该请求序列里的位置**（0-based，就是 page_table 的第 1 维下标）。
- **值 `page_table[req, pos]`**：一个 **flat slot 行号**，指向 KV 池 `[num_pages*page_size, …]` 的某一行——那一行就存着「req 的第 pos 个 token」的 K/V 向量。

所以取 KV 就是两步（和 notebook 17 的 `physical_kv_cache[block_table]` **同一个高级索引 gather**）：

```
slots = page_table[req, :seq_len]     # pos 0..seq_len-1  ->  一串 flat slot
kv    = kv_pool[slots]                # 高级索引 gather，拿到这条序列的全部 KV
```

写新 token 的 KV 也一样：`out_loc = page_table[input_mapping]`（`scheduler/scheduler.py:210`）先 gather 出新 token 该写的 slot，再 `store_cache(..., indices=out_loc)` 写进池子。

### 2. 树和 page_table 怎么联动（命中前缀时）

`match_prefix` 命中后，`RadixCacheHandle.get_matched_indices()`（`radix_cache.py:91`）沿 **leaf→root** 把各节点的 `value` `torch.cat` 起来，得到共享前缀的 slot 串；`prefill.py:61` 把它 `copy_` 进这个请求的 `page_table[req, :cached_len]`。→ 请求**直接复用**这些物理 slot，跳过 prefill。**树管「共享/淘汰」，page_table 给 kernel「每请求连续视图」，二者并存**。

### 3. 驱逐后内存怎么回收

- `RadixPrefixCache.evict(size)`（`radix_cache.py:148`）：收 `ref_count==0` 的叶子 → `heapq` 按 `timestamp` LRU → 弹最旧叶子、收集其 `node.value`、从父节点删除；**父节点若变叶子且未锁则继续弹**（`:172`，**leaf-first 自底向上**）→ `return torch.cat(evicted_indices)`。
- `CacheManager._allocate`（`cache.py:106`）：`free_slots = cat([free_slots, evicted[::page_size]])`——**淘汰出的 slot 索引拼回 free list**。物理池不动，`ref_count>0` 的节点锁住不淘汰（root 恒 `ref_count=1`）。

### 4. `page_size` 对齐分裂（回答「page_size=16 怎么 split」）

`radix_cache.py` 两行是关键：

```python
match_len = align_down(match_len, self.page_size)   # :219  匹配长度向下取整到页边界
node = node.split_at(match_len)                     # :224  在页对齐位置切
```

`node._key/_value` 始终是 **token 级**数组（`set_key_value` 断言 `len(key)==len(value)`），`split_at` 把它们在**页边界**（`page_size` 的倍数）处切成 `[:pos]`/`[pos:]`——切点正好落在页与页之间，**物理 page 从不被切开**。代价：页内的部分共享前缀被放弃（`page_size=1` 无此损失）。下面 demo 用同两条序列对比 `page_size=1` vs `16`。

```python
# 延伸 Demo：page_table -> slot -> KV pool，以及 page_size=1 vs 16 的 split
# （逻辑忠实照搬 mini-sglang radix_cache.py：key_fn / get_match_len / align_down / split_at / _tree_walk）
import torch


# ---------- Part A: page_table[req, pos] 怎么定位 KV ----------
print("=== Part A: page_table[req, pos] -> flat slot -> gather KV ===")
num_slots, head_dim = 12, 2
kv_pool = torch.arange(num_slots * head_dim, dtype=torch.float32).reshape(num_slots, head_dim)  # 固定物理池

page_table = torch.full((2, 8), -1, dtype=torch.int64)   # [max_reqs, max_len]
page_table[0, :4] = torch.tensor([7, 3, 9, 1])           # req0 的 4 个 token 落在 slot 7,3,9,1（乱序也行）

slots = page_table[0, :4]         # pos = 序列内位置(dim1)，取 0..3
kv = kv_pool[slots]               # 高级索引 gather == notebook 17 的 physical_kv_cache[block_table]
print("  page_table[0,:4] (pos->slot) =", slots.tolist())
print("  kv_pool[slots].shape         =", tuple(kv.shape), " (拿到这条序列的完整 KV)")
assert torch.equal(kv[0], kv_pool[7]) and torch.equal(kv[2], kv_pool[9])


# ---------- Part B: page_size=1 vs 16 的分裂 ----------
def align_down(x, p): return (x // p) * p
def get_key_fn(ps):   return (lambda x: x[0]) if ps == 1 else (lambda x: tuple(x[:ps]))
def get_match_len(key, ids):
    n = min(len(key), len(ids)); i = 0
    while i < n and key[i] == ids[i]: i += 1
    return i

class Node:
    def __init__(self, key_fn):
        self.key_fn = key_fn; self.children = {}; self._parent = None; self._key = []; self._value = []
    @property
    def length(self): return len(self._key)
    def set_kv(self, k, v):
        assert len(k) == len(v); self._key, self._value = k, v     # 源码不变量: key 与 value 等长
    def set_parent(self, p):
        self._parent = p; p.children[self.key_fn(self._key)] = self
    def split_at(self, pos):
        assert 0 < pos < self.length
        parent = self._parent
        mid = Node(self.key_fn)
        mid.set_kv(self._key[:pos], self._value[:pos])   # 前缀 -> mid
        mid.set_parent(parent)
        self.set_kv(self._key[pos:], self._value[pos:])  # 后缀 -> 旧节点
        self.set_parent(mid)
        return mid

class Radix:
    def __init__(self, page_size):
        self.page_size = page_size; self.key_fn = get_key_fn(page_size); self.root = Node(self.key_fn)
    def _walk(self, ids):
        plen, node = 0, self.root
        while plen < len(ids):
            child = node.children.get(self.key_fn(ids[plen:]))
            if child is None: return node, plen
            node = child
            m = align_down(get_match_len(node._key, ids[plen:]), self.page_size)   # 页对齐！
            plen += m
            if m != node.length:
                return node.split_at(m), plen                                     # 页边界处切
        return node, plen
    def insert(self, ids, val):
        insert_len = align_down(len(ids), self.page_size)
        ids, val = ids[:insert_len], val[:insert_len]
        node, plen = self._walk(ids)
        if plen != insert_len:
            leaf = Node(self.key_fn); leaf.set_kv(ids[plen:], val[plen:]); leaf.set_parent(node)
        return plen
    def dump(self):
        lines = []
        def dfs(n, d):
            if n is not self.root:
                lines.append("  " * d + f"key[{n._key[0]}..{n._key[-1]}] len={n.length}")
            for c in n.children.values(): dfs(c, d + 1)
        dfs(self.root, 0); return "\n".join(lines)

print("\n=== Part B: 同两条序列(实际共享 20 token)，page_size 不同 -> split 位置不同 ===")
A = list(range(32));                          Aslots = list(range(100, 132))
B = list(range(20)) + list(range(900, 912));  Bslots = list(range(200, 232))
for ps in (1, 16):
    r = Radix(ps); r.insert(A, Aslots); shared = r.insert(B, Bslots)
    print(f"\n page_size={ps}: 报告共享前缀 = {shared} token")
    print(r.dump())

r1 = Radix(1);  r1.insert(A, Aslots);  s1 = r1.insert(B, Bslots)
r16 = Radix(16); r16.insert(A, Aslots); s16 = r16.insert(B, Bslots)
assert s1 == 20 and s16 == 16
print("\n✅ page_size=1 在真实分叉点(20)切；page_size=16 只能在页边界(16)切 -> tokens 16..19 的共享被放弃")

```
