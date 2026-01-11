import numpy as np

def normalize(X):
    """将点投影回单位球面 S^{n-1}"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # 避免除以0
    return X / (norms + 1e-16)

def compute_gram_matrix(X):
    """计算格拉姆矩阵 G = X @ X.T (即余弦相似度)"""
    # 限制数值范围在 [-1, 1] 防止 acos 出现 NaN
    return np.clip(X @ X.T, -1.0, 1.0)

def project_forces(X, F):
    """
    将欧氏空间的力 F 投影到 X 处的切空间。
    F_tangent = F - (F . X) * X
    """
    # 计算径向分量 (每个粒子一个标量)
    radial_comp = np.sum(F * X, axis=1, keepdims=True)
    # 移除径向分量
    return F - radial_comp * X

def get_min_angle_deg(X):
    """辅助函数：计算当前最小夹角（度） 和 当前小于60度的对数"""
    G = compute_gram_matrix(X)
    np.fill_diagonal(G, -1.0) # 排除自身
    max_cos = np.max(G)
    min_angle = np.arccos(max_cos)
    return np.degrees(min_angle), np.sum(G > np.cos(np.deg2rad(60)))

def exp_map_single(x, v):
    """
    x : (d,) unit vector
    v : (d,) tangent vector (x·v = 0)
    """
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return x
    return np.cos(theta) * x + np.sin(theta) / theta * v

def find_min_angle_pair(X):
    G = X @ X.T
    np.fill_diagonal(G, -1.0)
    i, j = np.unravel_index(np.argmax(G), G.shape)
    min_angle = np.arccos(G[i, j]) * 180 / np.pi
    return i, j, min_angle

