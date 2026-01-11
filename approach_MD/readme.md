# Spherical Point Optimization

本项目用于在高维球面上（S^{dim-1}）优化 N 个点的位置，使得点之间的最小夹角尽可能大。当前实现基于 **Riesz 势能 + 球面动力学优化**，并采用分阶段（curriculum）策略逐步逼近高质量构型。

---

## 一、代码结构

```text
.
├── main.py          # 主入口文件（命令行接口）
├── geometry.py      # 几何相关工具（归一化、最小夹角等）
├── potential.py     # 势能定义（RieszPotential 等）
├── dynamics.py      # 球面动力学优化器（SphericalOptimizer）
└── README.md        # 使用说明（本文档）
```

---

## 二、运行环境

* Python >= 3.8
* 依赖库：

  * numpy

（当前代码不依赖 scipy / torch 等第三方库）

---

## 三、基本思想简介

* 每个点表示为 (\mathbb{R}^{dim}) 中的单位向量
* 点间夹角由内积决定：
  [ \cos \theta_{ij} = x_i \cdot x_j ]
* 优化目标：**最大化最小夹角**（等价于避免点过近）
* 实现方式：

  * 使用 Riesz 势能作为排斥力
  * 在球面切空间中进行动力学更新
  * 通过多阶段势能指数逐步“变硬”

---

## 四、命令行接口（main.py）

### 1. 基本用法

```bash
python main.py --dim DIM --N N [--save_path PATH] [--seed SEED] [--steps STEPS]
```

### 2. 参数说明

| 参数名           | 类型  | 是否必须 | 含义                     |
| ------------- | --- | ---- | ---------------------- |
| `--dim`       | int | 是    | 球面的嵌入维数（点位于 S^{dim-1}） |
| `--N`         | int | 是    | 球面上的点数                 |
| `--save_path` | str | 否    | 保存最终构型的路径（.npy 文件）     |
| `--seed`      | int | 否    | 随机种子（默认 43）            |
| `--steps`     | int | 否    | 最后阶段运行步数         |

---

### 3. 示例

#### 示例 1：5 维球面，40 个点，不保存结果

```bash
python main.py --dim 5 --N 40
```

#### 示例 2：保存最终构型

```bash
python main.py --dim 5 --N 40 --save_path data/dim5_N40.npy
```

---

## 五、输出说明

程序运行过程中会打印：

* 初始最小夹角
* 每个 stage 的进度
* 每 1000 步的监控信息：

  * 当前最小夹角（单位：度）
  * 小于 60° 的点对数量


## 六、结果文件说明（.npy）

若指定 `--save_path`，程序会保存一个 NumPy 数组：

```python
X.shape == (N, dim)
```

其中：

* 每一行是一个单位向量
* 表示球面上的一个点