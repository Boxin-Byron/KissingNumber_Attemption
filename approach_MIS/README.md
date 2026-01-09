# MIS：最大点独立集（MIS）求 Kissing Number 的下界

这个目录实现了一个**离散化 + 图论**的 kissing number 下界搜索流程：

- 在 $\mathbb{R}^d$ 中，候选球心位于半径为 2 的 $S^{d-1}$ 上（单位球相切球心的正确半径）。
- 将“球心距离过近（\<2）”等价转换为冲突图的边。
- 用 **Maximum Independent Set (MIS)**（默认用 **Gurobi** 精确求解）得到一组互不冲突的球心集合，作为 $K(d)$ 的一个**合法下界**。
- 为了更容易命中大解，支持：
	- **FPS（Farthest Point Sampling）**：从大规模均匀采样中筛出覆盖更均匀的候选点；
	- **soft→hard**：先用放松约束的“soft 图”求一个大的 MIS，再通过连续优化式的“推远 + 投影回球面”修复把最小距离尽量拉回 2（最终必须通过 hard 校验才算下界）。



## 理论原理

### 1) 采样与 FPS（让候选集合更像“覆盖”）

- `src/sampling.py::sample_sphere_uniform`：通过高斯向量归一化在 $S^{d-1}$ 上均匀采样，并缩放到半径 2。
- `src/sampling.py::sample_sphere_fps`：先 oversample，再用 **FPS** 贪心选点，最大化“到已选集合的最小距离”，得到更均匀的覆盖（避免候选点扎堆）。

FPS 的直观目标是“最大化最小间距”：

$$
\max \ \min_{i\ne j}\|x_i-x_j\|
$$

### 2) 建图（soft/hard 阈值不同）

- `src/graph.py::build_conflict_graph(points, min_dist, epsilon, backend)`

把每个候选球心当作一个点 $v_i$。

- **Hard（物理正确）冲突条件**：如果 $\|v_i-v_j\| < 2$，两球会重叠，因此连边。
- **Soft**：把阈值改成

$$
	soft\_min = 2\cdot(1-\texttt{soft\_delta})
$$

soft 的 MIS 结果本身不一定 hard 合法，因此不能直接当下界；它更像一个“seed”。

### 3) MIS（用 Gurobi 精确求最大独立集）

- `src/mis_solver.py::solve_mis`

标准 0-1 ILP：

$$
\max\sum_{i\in V} x_i
$$

$$
	s.t.\ \ x_i + x_j \le 1,\ \forall (i,j)\in E;\ \ x_i\in\{0,1\}
$$

当前代码里也包含 `greedy` 作为快速近似。

### 4) soft→hard（先放松找大集合，再修复到严格可行）

- `src/soft_hard.py::solve_soft_then_repair`：封装 soft MIS + repair 多次尝试的流程。
- `src/refine.py::refine_repulsion_projected`：连续的“排斥/推远”更新，并始终投影回半径 2 的球面。

最终必须通过 hard 校验：

$$
\min_{i<j}\|x_i-x_j\| \ge 2 - \texttt{validate\_tol}
$$

只有通过 hard 校验的最终结果，才是 kissing number 的有效下界。

## 目录结构

```
approach_MIS/
├── README.md
├── requirements.txt
├── src/
│   ├── sampling.py           # uniform + FPS
│   ├── graph.py              # 冲突图构建（KDTree）
│   ├── mis_solver.py         # Gurobi 求 MIS
│   ├── soft_hard.py          # soft→hard 管线封装（soft MIS + repair）
│   ├── refine.py             # 连续推远 + 投影回球面（repair 核心）
│   └── visualize.py          # 可视化（2D/3D配置 + 距离分布等）
├── experiments/
│   ├── run_2d.py              # 2D 实验入口
│   ├── run_3d.py              # 3D 实验入口
│   ├── run_4d.py              # 4D 实验入口
│   ├── run_5d.py              # 5D 实验入口
│   └── scan_greedy_seeds.py   # 扫描种子/启发式脚本
└── outputs/
		├── 2d/ 3d/ 4d/ 5d/        # 图片与summary等输出
```

## 实验结果

| 维度 d | soft条件下结果 | n_candidates | sampling | soft_delta | min_distance |
|---:|---:|---:|---|---:|---|
| 2 | 6 | 1000 | 500000 | 0.002 | 1.9995 |
| 3 | 12 | 1000 | N/A | N/A | 2.0021 |
| 4 | 24 | 3000 | 500000 | 0.06 | 1.9951 |
|  | 23 | 5000 | 100000 | 0.04 | 1.9983 |
| 5 | 37 | 5000 | 1000000 | 0.10 | 1.9413 |
|  | 33 | 5000 | 500000 | 0.08 | 1.9885 |
|  | 32 | 5000 | 500000 | 0.06 | 1.9853 |

注：三维Kissing Number无需FPS和soft/hard即可实现12

## 推荐参数（2D-5D）

- 2D：`python .\experiments\run_2d.py -n 1000 -s uniform --soft-delta 0.002 --repair-steps 2000 --repair-step-size 0.02 --repair-restarts 30 --validate-tol 1e-3`
- 3D：`python .\experiments\run_3d.py`（默认参数即可）
- 4D：`python .\experiments\run_4d.py -n 5000 -m auto -t 600 --seed 42 --sampling fps --oversample 500000 --fps-start random --soft-delta 0.06 --repair-steps 8000 --repair-step-size 0.03 --repair-restarts 10 --repair-stage-mins "1.97,1.985,1.993,1.997,2.0" --repair-stage-fracs "0.05,0.10,0.20,0.25,0.40" --validate-tol 1e-3`
- 5D：`python .\experiments\run_5d.py -n 5000 -m auto -t 3600 --seed 42 --sampling fps --oversample 500000 --fps-start random --soft-delta 0.1 --repair-steps 8000 --repair-step-size 0.03 --repair-restarts 10 --repair-stage-mins "1.97,1.985,1.993,1.997,2.0" --repair-stage-fracs "0.05,0.10,0.20,0.25,0.40" --validate-tol 1e-3`

## 实验总结

* 引入FPS采样后，对3维和4维的初始解有很大提升。FPS对高维未必有效，有待验证。

* 由于方法本身的特殊性，对于2维和4维，必须引入soft/hard对点的位置进行调整，现有方法仍无法将min_distance提升到2-(1e-4)的水平。

* 在5维引入soft/hard也对找到好的初始解有很大帮助，但在过大soft_delta下解的有效性损失。例如求解出的37在进一步调整前很难保证有效。

* 在一定范围内增大n, oversample都有提升作用。增大前者会导致优化问题求解极慢。

* 结果高度依赖采样本身，随机性强。修改random_seed会带来较大变动。后续可考虑粗略扫描一遍random_seed，有较小提升。

* 后续的repair阶段分多阶段提升不明显，但暂且保留。

## 后续

* MIS为NP hard，在点数多时求解速度非常缓慢。但是因为思路本身要求用随机撒点来近似最优结构，在高维空间需要极大的点数。会导致方法非常低效。

* 对于2维，4维的精确解。思路本身导致采样到解的概率测度意义上为0。soft/hard需要工程的进一步改进，使结果更接近2，否则难以保证解的有效性。

* 求解MIS在gurobi产生的heuristic solution的基础上提升有限。一般只能提升3-4。前面提到的改进大多也直接反映到heuristic solution上。

* 综上所述。本思路不适合直接推广至更高维度进行计算。可以继续探讨FPS等启发式采样+贪心算法能否快速在高维得到较好初始解，对位置进行调整找出更多可行结构。