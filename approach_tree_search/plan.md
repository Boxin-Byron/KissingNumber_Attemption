# Approach 3: Geometric Construction + Tree Search

## 1. 问题定义 (Problem Definition)
目标：通过几何构造和搜索算法，逐步构建 $d$ 维空间中的 Kissing Number 构型。
利用几何性质：如果一个新球与 $d$ 个已有球相切，其位置可以通过解方程组确定。

## 2. 核心思路 (Core Idea)
将寻找最大 Kissing Number 建模为树搜索问题。
*   **状态 (State)**: 当前已放置的互不重叠的球的集合。
*   **动作 (Action)**: 选择 $d$ 个已有球，计算与它们及中心球相切的新球位置（候选点）。
*   **搜索 (Search)**: 使用树搜索策略（如 Beam Search 或 MCTS）来探索状态空间，寻找包含球数最多的最终状态。

## 3. 解决步骤 (Implementation Steps)
1.  **几何求解器 (Geometric Solver)**:
    *   实现根据 $d$ 个已知球心，求解与它们及原点距离均为 2 的新点坐标。
    *   处理数值误差和无解情况。

2.  **合法性检查 (Validity Check)**:
    *   检查新生成的球是否与当前集合中其他球重叠（距离 < 2 - epsilon）。

3.  **搜索算法 (Search Algorithm)**:
    *   **初始化**: 放入初始种子球（Seed Spheres）。
    *   **扩展 (Expand)**: 对当前状态，采样不同的 $d$-tuple 组合，生成候选新球。
    *   **评估与剪枝 (Evaluate & Prune)**: 评估状态潜力（如当前球数、空隙大小），保留最有希望的分支。
    *   **回溯/迭代**: 直到无法添加新球。

4.  **策略网络 (Policy Network - Optional/Later)**:
    *   如果纯搜索太慢，可以训练一个简单的 MLP 预测哪个 $d$-tuple 更容易产生合法解（本项目初期先用启发式规则）。

## 4. 文件架构设计 (File Structure)
```
approach_3_tree_search/
├── plan.md                 # 本计划文件
├── README.md               # 运行说明
├── src/
│   ├── __init__.py
│   ├── geometry.py         # 几何求解与碰撞检测 (Solver & Collision)
│   ├── search.py           # 树搜索逻辑 (Tree Search / Beam Search)
│   ├── state.py            # 状态定义 (State Representation)
│   └── visualize.py        # 可视化工具
└── experiments/
    ├── run_2d_search.py    # 2D 验证
    ├── run_3d_search.py    # 3D 验证
    └── run_4d_search.py    # 4D 探索
```
