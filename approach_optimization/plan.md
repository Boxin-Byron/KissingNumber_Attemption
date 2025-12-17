# Approach 1: Continuous Optimization (Gradient Descent)

## 1. 问题定义 (Problem Definition)
目标：在 $d$ 维空间中，寻找最大数量 $N_{max}$ 的单位球，使得它们：
1. 与中心单位球相切（即球心距离原点为 2）。
2. 彼此不重叠（即任意两球心距离 $\ge 2$）。

虽然最终目标是找到 $N_{max}$，但梯度下降算法需要预先固定球的数量 $N$。因此，我们的策略是：**固定 $N$ 进行优化判定，通过外层循环寻找最大的可行 $N$**。

## 2. 核心思路 (Core Idea)
采用 **"试探-验证" (Probe and Verify)** 策略：
1.  **内层循环 (Optimization)**: 给定一个固定的 $N$，利用梯度下降优化球的位置，尝试将 Loss 降为 0。如果 Loss $\approx 0$，则说明 $N$ 个球可以放入。
2.  **外层循环 (Search Strategy)**: 从已知的下界开始，逐步增加 $N$。对于每个 $N$，多次运行内层优化（因为非凸优化容易陷入局部最优）。如果某次运行成功，则记录结果并尝试 $N+1$；如果多次尝试均失败，则停止或认为当前 $N$ 已接近极限。

## 3. 解决步骤 (Implementation Steps)
1.  **初始化 (Initialization)**:
    *   设定维度 $d$。
    *   设定起始数量 $N_{start}$ (例如已知下界)。

2.  **内层优化 (Inner Loop - Fixed N)**:
    *   随机初始化 $N$ 个向量。
    *   **Loss Function**: $L = \sum_{i<j} \max(0, 2 - \|x_i - x_j\|)^2$。
    *   使用 Adam 优化器迭代更新坐标。
    *   **判定**: 检查最终构型是否存在重叠。

3.  **外层搜索 (Outer Loop - Find Max N)**:
    *   令当前尝试数量 $N = N_{start}$。
    *   运行内层优化 $K$ 次（随机重启）。
    *   如果至少有一次成功（无重叠）：
        *   保存结果。
        *   $N \leftarrow N + 1$，继续尝试。
    *   如果 $K$ 次全部失败：
        *   认为当前 $N$ 难以实现，输出上一个成功的 $N$ 作为结果。

4.  **验证与后处理 (Validation)**:
    *   统计最终有多少对球仍然重叠。
    *   移除重叠严重的球，计算剩余有效球的数量。

## 4. 文件架构设计 (File Structure)
```
approach_1_optimization/
├── plan.md                 # 本计划文件
├── README.md               # 运行说明
├── src/
│   ├── __init__.py
│   ├── optimize.py         # 核心优化循环 (Optimization Loop)
│   ├── loss.py             # 损失函数定义 (Loss Functions)
│   ├── utils.py            # 几何计算工具 (Geometry Utils)
│   └── visualize.py        # 2D/3D 可视化 (Visualization)
└── experiments/
    ├── run_2d.py           # 2D 实验脚本 (N=6)
    ├── run_3d.py           # 3D 实验脚本 (N=12)
    └── find_max_n.py       # 自动搜索最大 N 的脚本 (核心逻辑)
```
