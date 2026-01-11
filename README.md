# Kissing Number Problem - 方法总览

本项目包含四种不同的方法来探索和求解接吻数问题（Kissing Number Problem）。每种方法都在独立的目录下，通过不同的算法视角尝试寻找球面上最大不重叠球体数量。

## 目录结构

*   `approach_MIS/`: **最大独立集 (Maximum Independent Set) 方法**
*   `approach_optimization/`: **连续优化 (Continuous Optimization) 方法**
*   `approach_tree_search/`: **树搜索与几何构造 (Tree Search) 方法**
*   `approach_MD/`: **分子动力学 (Molecular Dynamics) 方法** *(注：如目录缺失，可能为历史版本或待集成)*

---

## 1. Approach MIS: 基于图论的最大独立集

该方法通过在球面上随机采样大量点，建立“冲突图”（如果两个点太近则连边），然后求解该图的最大独立集（MIS）。这是一种离散化近似方法。

*   **核心逻辑**: 随机采样 -> 冲突检测 -> Gurobi/Heuristic 求解 MIS -> 局部修正 (Repair)。
*   **适用场景**: 低维度验证，或者作为寻找下界的随机基准。
*   **代码路径**: `approach_MIS/`

### 运行方式

进入对应目录运行实验脚本（以 4D 为例）：

```bash
# 运行 4D 实验，采样 5000 个候选点
python approach_MIS/experiments/run_4d.py --candidates 5000 --method auto
```

参数说明：
*   `--candidates`: 采样点数
*   `--method`: MIS 求解器 (auto/gurobi)

---

## 2. Approach Optimization: 连续优化

该方法将问题建模为连续空间中的势能最小化问题。使用梯度下降（Gradient Descent）来最小化球体间的重叠量（Soft Overlap Loss）。

*   **核心逻辑**: 随机初始化 N 个点 -> 定义损失函数 (Overlap) -> Adam/SGD 优化 -> 检查是否满足硬约束。
*   **适用场景**: 寻找紧致结构，适合利用 GPU 加速并行搜索。
*   **代码路径**: `approach_optimization/`

### 运行方式

最常用的脚本是自动搜索最大 N 的脚本：

```bash
# 尝试在 3D 中寻找最大 N
python approach_optimization/experiments/find_max_n.py
```

或者运行特定维度的固定实验：
```bash
python approach_optimization/experiments/run_3d.py
```

---

## 3. Approach Tree Search: 树搜索与几何构造

该方法使用构造式算法 (Constructive Algorithm) 结合束搜索 (Beam Search)。它不是随机撒点，而是利用几何约束，在现有球体的“凹槽” (Slots) 中逐个添加新球体。

*   **核心逻辑**: 如果有 d-1 个球体两两接触，则可以计算出与其相切的候选位置 -> 束搜索维护最佳的 K 个构型 -> 结合“松弛” (Relaxation) 和“晶格偏置” (Lattice Bias) 优化。
*   **特点**: **最新改进版**。集成了模拟退火/SGD 松弛策略以及针对 PackerStar 论文的晶格优化思路。
*   **代码路径**: `approach_tree_search/`

### 运行方式

使用通用的搜索脚本 `run_search.py`：

```bash
# 运行 4D 搜索，束宽 (Beam Width) 设为 100
python approach_tree_search/experiments/run_search.py --dim 4 --beam 100
```

---

## 4. Approach MD: 分子动力学 (Molecular Dynamics)

该方法基于物理模拟，使用 Riesz 能量势（Riesz Potential）模拟粒子在球面上的相互排斥运动，随着“冷却”过程达到能量最低态（即最均匀分布）。

*   **核心逻辑**: 定义 Riesz 势能 -> 模拟退火/动力学演化 -> 逐渐硬化势能逼近硬球模型。
*   **代码路径**: `approach_MD/` *(如果存在)*

### 运行方式

```bash
# 在 5D 空间模拟 40 个粒子
python approach_MD/main.py --dim 5 --N 40
```

---

## 环境要求

所有方法均依赖以下基础库（建议在 `sparc` conda 环境下运行）：

*   Python 3.8+
*   NumPy, SciPy
*   PyTorch (CUDA 推荐)
*   NetworkX
*   Gurobi (可选，仅 Approach MIS 需要)
