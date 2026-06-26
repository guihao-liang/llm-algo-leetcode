# 使用指南

本页只回答两件事：内容先在哪看，Notebook 该在哪跑。

先把规则说清楚：

- `Colab` 是阅读和运行的入口，不是单独的环境层。
- `CPU-first` 和 `GPU-required` 是执行能力，不是入口名称。
- `CNB` / `Docker` / 云端 GPU 是统一交付方式，不是新的内容分层。

## 统一规则

| 场景 | 默认环境 | 备注 |
|---|---|---|
| 只是阅读内容 | 在线站点 | 不需要本地环境 |
| Part 0 / Part 1 | 在线 Notebook 或本地基础环境 | 主要是 Python、Jupyter、NumPy |
| Part 2 | 本地 CPU-first / Colab CPU / CNB CPU | 大多数题可在 CPU 上完成，少数题再切 GPU |
| Part 3 | 本地 NVIDIA GPU / Colab GPU / CNB GPU | 完整验收需要 GPU |
| Part 4 | 本地 NVIDIA GPU / Colab GPU / CNB GPU | 完整验收需要 GPU |
| 团队统一交付 | CNB / Docker / 云端 GPU | 适合一致性和复现 |

如果你只想记一句话：

**先在线阅读，再按 Part 0-2 / Part 3-4 选择 CPU-first 或 GPU-required 环境；CNB 和 Docker 负责统一交付。**

## 环境选择顺序

如果你不知道先选什么环境，按这个顺序判断：

1. 只是先看内容
   - 直接用在线站点
   - 先看 Part 0 / 1 的导学页

2. 要做 Part 0 / 1
   - 优先在线 Notebook
   - 也可以用本地基础环境
   - 目标是先把 Python、Jupyter、NumPy 跑通

3. 要做 Part 2
   - 默认走本地 CPU-first
   - 也可以用 Colab CPU 或 CNB CPU
   - 统一一套 Python 依赖
   - 少数题才需要切 GPU

4. 要做 Part 3
   - 需要 GPU 环境
   - 可以用本地 NVIDIA GPU、Colab GPU 或 CNB GPU
   - 如果当前会话没有 GPU，就只能阅读，不能按完整路径验收

5. 要做 Part 4
   - 同样需要 GPU 环境
   - 可以用本地 NVIDIA GPU、Colab GPU 或 CNB GPU
   - 重点是 CUDA / system tooling 的完整验证

6. 要给团队/课程统一环境
   - 优先 CNB / Docker
   - 目标是“大家按同一套环境跑”

如果还是拿不准，默认顺序是：

**在线阅读 -> Part 0 / 1 在线 Notebook -> Part 2 CPU-first -> Part 3 / 4 GPU-required -> CNB / Docker 统一交付**

## 执行层与交付方式

### 1. 轻量学习层

适用部分：
- Part 0
- Part 1

环境特点：
- Python
- Jupyter / Notebook
- NumPy / 基础科学计算
- Colab / 在线 Notebook 可直接开始

这两章对部署要求低，重点是把基础概念和前置知识学稳，不需要一开始就上复杂容器或 GPU 工具链。

### 2. 主力学习层

适用部分：
- Part 2

环境特点：
- 统一 Python 环境
- PyTorch
- correctness 验证脚本
- 本地 CPU 足够覆盖大多数题

- Part 2 是 **CPU-first**
- 已确认至少 `21_Gradient_Checkpointing` 需要 NVIDIA GPU 才能测真实 CUDA 显存峰值
- 学习时建议尽量使用同一套 Python 依赖，保证练习和验证结果一致
- 如果不想手动配本地环境，也可以直接用 CNB 作为第二选择

### 3. 高门槛实验层

适用部分：
- Part 3

环境特点：
- Linux
- NVIDIA GPU
- CUDA
- Triton
- 编译工具链

- Part 3 是 **GPU-required**
- 少数页面可能有阅读路径或 CPU fallback，但不构成完整学习路径

如果你在 Colab 中打开 Part 3，请优先选择免费的 `T4 GPU`，或者任意可用的 GPU runtime，再运行 notebook 最前面的环境准备单元。该单元会在 `triton` 缺失时自动安装依赖；如果连 GPU runtime 都没有，Part 3 只能阅读，不能按完整路径验收。

Part 3 的主叙事链是 `PyTorch -> Triton`：先在 PyTorch 层理解算法与行为，再用 Triton 把算子高效落到 GPU。

Part 4 则承接 `CUDA -> System -> Architecture`：继续向下处理 kernel、通信、ZeRO 和技术选型。

### 4. 统一交付方式

适用场景：
- 团队协作
- 统一实验镜像
- 云端开发
- 需要减少本地环境差异时

平台选项：
- Colab
- CNB
- Docker
- 云端 GPU 托管环境

这层主要服务 Part 2 后段、Part 3 和 Part 4，目标是把“能跑”变成“大家都按同一套环境跑”。

CNB 在这里的角色是“统一交付环境”，不是“学习内容入口”。

当前验证状态：
- **Colab**：已验证，可一键直达
- **Docker**：推荐以 **50 系 NVIDIA GPU** 作为当前已验证基线；40 系和 30 系后续再单独补充兼容矩阵与验证结果
- **CNB**：分成两层看，`push` / `pull_request` 负责流水线校验，`vscode` 负责读者打开后的交互环境；当前已完成仓库同步，交互环境仍需按 CPU/GPU 会话分别验证
- **ModelScope / 其他在线 Notebook**：待验证

## 操作系统和本地部署

本项目的本地部署不是只看“能装上依赖”，还要看操作系统、包管理方式和测试脚本是否统一。

### 操作系统差异

- **Linux 22.04**：当前已验证的主基线，优先作为本地开发和 Part 2 / 3 / 4 验证环境
- **WSL2**：可作为过渡方案，但尚未作为正式主基线承诺
- **macOS**：适合轻量阅读和部分 Python 逻辑练习；Part 3 / Part 4 的完整 GPU 体验不做默认承诺
- **Windows 原生环境**：不作为主支持路径，若要学习建议优先走 WSL2、CNB 或在线 Notebook

### conda / venv 选择

- **conda 是推荐主路径**：本项目的 `environment.yml`、`cnb/environment.yml` 都以 conda 为核心入口，便于统一 Python、PyTorch 和 GPU 依赖
- **venv 也可以用**：如果你只需要纯 Python 虚拟环境，可以用 venv，但需要手动按 `requirements/*.txt` 安装依赖
- **不建议混用**：同一份环境里不要同时把 conda、venv 和系统 Python 混着装，否则后面验证结果不稳定

### 依赖版本策略

- `requirements/base.txt`：基础版本来源
- `requirements/dev.txt`：开发和测试版本来源
- `requirements/gpu.txt`：Part 3 / Part 4 / Triton / CUDA 扩展版本来源
- `environment.yml`：只负责 Python 版本和把上述依赖串起来
- `cnb/environment.yml`：CNB 侧的统一环境骨架

更新依赖时，优先改 `requirements/*.txt`，再同步 `environment.yml` 和 `cnb/environment.yml`，避免出现“本地一套版本、云端另一套版本”的情况。

### 测试脚本的位置

测试脚本是本地环境部署的一部分，不只是“可选附加项”。

- `verify.py`：统一维护入口，优先用于日常验证和部分级回归
- `project_test_scripts.md`：测试脚本索引
- `test_chapter0_1_notebooks.py`：Part 0 / 1 的顺序执行验证
- `test_notebook_answers.py`：Part 2 / 3 / 4 的答案区验证
- `check_chapter_links.py`：站内链接检查

如果环境已经装好，但测试脚本跑不起来，通常说明依赖版本、OS 差异或 GPU 兼容性还没有收口。

这里也是一条**二选一且有优先级**的关系：
- 日常回归先跑 `verify.py`
- 只在排查单个 notebook、单个目录或底层脚本实现时，才直接跑旧脚本

`verify.py` 的常用入口是：

```bash
python verify.py chapter0_1
python verify.py chapter2
python verify.py chapter3
python verify.py chapter4
python verify.py all
```

默认会做转换、镜像检查和章节级验证；加 `--no-build` 可以跳过 docs 构建。

### CNB 里怎么验证

CNB 的验证顺序和本地类似，但要先确认当前会话拿到的是项目环境，而不是系统 Python。

#### 1. 先检查交互环境

```bash
. /opt/conda/etc/profile.d/conda.sh
conda activate llm_algo_cnb_dev

python --version
python -m pip --version
python -c "import torch; print(torch.__version__)"
python -c "import triton; print(triton.__version__)"
```

如果这些命令都能正常执行，说明 CNB 交互环境已经进入项目的统一软件栈。

#### 2. 再验证 Part 0 / 1

```bash
python verify.py chapter0_1 --no-build
```

如果需要定点排查某个 Part 0 / 1 notebook，再直接跑：

```bash
python test_chapter0_1_notebooks.py
```

这一步用于确认基础 notebook、依赖和顺序执行链路没有问题。

#### 3. 再验证 Part 2

```bash
python verify.py chapter2 --no-build
```

如果需要绕过统一入口做底层排查，再直接跑：

```bash
python test_notebook_answers.py --all --dir 02_PyTorch_Algorithms --mode both
```

Part 2 是 CNB 里最应该优先跑通的主链路。

#### 4. Part 3 / Part 4 只在有 GPU 时做完整验证

```bash
python verify.py chapter3 --no-build
python verify.py chapter4 --no-build
```

如果需要绕过统一入口做底层排查，再直接跑：

```bash
python test_notebook_answers.py --all --dir 03_Triton_Kernels --mode both
python test_notebook_answers.py --all --dir 04_CUDA_and_System_Optimization --mode both
```

如果当前 CNB 实例没有 GPU，`torch.cuda.is_available()` 会是 `False`，这时不要把 Part 3 / Part 4 的 GPU 结果当成最终验收。

### Part 3 / Part 4 的 GPU 入口

Part 3 / Part 4 需要单独的 GPU 验证入口，不建议和默认 CNB 交互环境混用。

- 默认 CNB 交互环境：面向 Part 0 / 1 / 2，主要验证 Python、Notebook 和 CPU 路径
- GPU 验证入口：面向 Part 3 / Part 4，必须使用平台分配到 GPU 节点的会话
- 对应的 CNB 入口名：`vscode-gpu`
- 验证脚本：`python verify.py chapter3 --no-build` / `python verify.py chapter4 --no-build`

如果当前会话没有 GPU，就只做：
- 环境检查
- 站内链接检查
- Part 3 / Part 4 的非 GPU 路径检查

不要把 `torch.cuda.is_available() == False` 的会话写成 Part 3 / Part 4 最终通过。

### Notebook 使用

- 执行单元格：`Shift + Enter` 或 `Ctrl + Enter`
- 执行所有单元格：`Run -> Run All Cells`
- 刷题流程：先看导学，再填 TODO，再跑测试
- 如果 notebook 报 `name not defined`，通常是前面的 cell 没按顺序执行

## Part 2 / 3 / 4 的实际要求

### Part 2

- 整体定位：CPU-first
- 大多数 notebook：可在 CPU 环境下完成学习和 correctness 验证
- 已确认 GPU 题：`21_Gradient_Checkpointing`
- 推荐策略：统一 Python 环境，GPU 用于后段实验和真实性能验证
- 已验证：当前本机 `llm_algo` 环境下，Part 2 答案区全量通过

### Part 3

- 整体定位：GPU-required
- 完整体验：Linux + NVIDIA GPU + CUDA / Triton
- 代码审计结果：本章直接面向 GPU 内核、显存和通信行为
- 推荐策略：把 Linux 作为默认参考环境，其他系统只作为补充
- 已验证：当前本机 `llm_algo` 环境下，Part 3 答案区全量通过

### Part 4

- 整体定位：GPU-required
- 完整体验：Linux + NVIDIA GPU + CUDA / system tooling
- 代码审计结果：本章直接面向 CUDA kernel、通信、系统优化和架构选型
- 推荐策略：把 Linux + GPU 作为默认参考环境，其他系统只作为补充
- 当前状态：已通过转换、镜像和入口链接检查；完整 GPU 答案区仍需在 GPU 环境补验

### 本地平台备注

- **Linux 22.04**：已验证，是当前最稳的本地参考环境
- **50 系 NVIDIA GPU**：已验证，是当前本机 GPU 学习与 Part 3 验证基线；Part 4 需要补充完整 GPU 验证
- **40 系 NVIDIA GPU**：暂未作为已验证基线，后续补充兼容矩阵
- **WSL2**：理论上可作为过渡方案，但尚未作为主验证基线
- **macOS**：可能接近可用，但尚未完成逐项验证，不应默认承诺 Part 3 / Part 4 完整体验

### 本地 GPU 兼容矩阵（当前）

| GPU 代际 | 当前状态 | 适用范围 | 备注 |
|---|---|---|---|
| 50 系 NVIDIA GPU | 已验证 Part 2 / 3，待补 Part 4 | Part 2 / Part 3 本地验证基线，Part 4 兼容目标 | Part 4 需要单独补 GPU 结果 |
| 40 系 NVIDIA GPU | 待补充验证 | Part 3 / Part 4 兼容目标 | 后续单独补版本与兼容结果，不默认承诺 |
| 30 系 NVIDIA GPU | 待补充验证 | Part 3 / Part 4 兼容目标 | 仅作为潜在兼容目标，需单独实测 |
