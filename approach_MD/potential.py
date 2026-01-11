import numpy as np
from geometry import compute_gram_matrix

class RieszPotential:
    def __init__(self, s_power: float, mode: str='sum'):
        """
        Riesz s-potential: V = 1 / r^s
        Force F_ij = s * r^(-s-2) * (X_i - X_j)
        
        参数:
        s_power: 斥力指数 s。
                 s=2 对应库仑力/万有引力 (长程)。
                 s 大 (如 15, 60) 对应硬球排斥 (短程)。
        """
        self.s = s_power
        self.mode = mode

    def compute_force(self, X):
        """
        计算粒子受到的总斥力。
        
        公式: F_i = sum_{j!=i} W_{ij} * (X_i - X_j)
        其中 W_{ij} 是与距离相关的标量权重。
        
        对于 Riesz 势 (1/r^s):
            W_{ij} = s * r_{ij}^{-(s+2)}
        由于 r_{ij}^2 = 2(1 - cos_theta)，即 r^2 = 2 * dist_cos
        """
        N, dim = X.shape
        G = compute_gram_matrix(X)  # G[i,j] = cos(theta)
        
        # 1. 计算距离项
        # 我们使用 1 - cos(theta) 作为距离的代理，它正比于 r^2
        # r^2 = 2 * (1 - cos)
        # 加上 1e-8 防止除零 (对角线或重合点)
        dist_sq_proxy = 1.0 - G
        np.fill_diagonal(dist_sq_proxy, 1.0) # 对角线设为非0值以免报错，后续会将权重设为0
        
        # 2. 计算权重矩阵 W_{ij}
        #weights = 1 / (np.exp(self.s * (dist_sq_proxy - 0.5)) + 1)
        weights = np.power(2 * dist_sq_proxy, -(self.s + 2))
        
        # 3. 处理对角线
        # 粒子对自己没有力，设为 0
        np.fill_diagonal(weights, 0.0)
        
        # 如果是max 模式，则每行只保留最大的权重，其他设零
        if self.mode == 'max':
            max_indices = np.argmax(weights, axis=1)
            new_weights = np.zeros_like(weights)
            for i in range(N):
                new_weights[i, max_indices[i]] = weights[i, max_indices[i]]
            weights = new_weights
        # 4. 计算合力 F_i = sum_j W_{ij} (X_i - X_j)
        # 展开为: F_i = (sum_j W_{ij}) * X_i - sum_j (W_{ij} * X_j)
        
        # 第一项: sum_weights * X
        # (N,) -> (N, 1) 用于广播
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        term1 = sum_weights * X 
        
        # 第二项: weights @ X
        # (N, N) @ (N, d) -> (N, d)
        term2 = weights @ X
        
        # 总力 (方向由 X_i 指向 X_j 的反方向，即排斥)
        F = term1 - term2
        
        return F

class FermiPotential:
    def __init__(self, s_power: float, mode: str='sum'):
        self.s = s_power
        self.mode = mode

    def compute_force(self, X):
        """
        计算粒子受到的总斥力。
        
        公式: F_i = sum_{j!=i} W_{ij} * (X_i - X_j)
        其中 W_{ij} 是与距离相关的标量权重。
        """
        N, dim = X.shape
        G = compute_gram_matrix(X)  # G[i,j] = cos(theta)
        
        dist_sq_proxy = 1.0 - G
        np.fill_diagonal(dist_sq_proxy, 1.0) # 对角线设为非0值以免报错，后续会将权重设为0
        
        weights = 1 / (np.exp(self.s * (dist_sq_proxy - 0.5)) + 1)
        
        np.fill_diagonal(weights, 0.0)
        
        if self.mode == 'max':
            max_indices = np.argmax(weights, axis=1)
            new_weights = np.zeros_like(weights)
            for i in range(N):
                new_weights[i, max_indices[i]] = weights[i, max_indices[i]]
            weights = new_weights
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        term1 = sum_weights * X 
        term2 = weights @ X
        F = term1 - term2
        
        return F

if __name__ == '__main__':
    dist_sq_proxy = np.linspace(1-np.cos(np.radians(58)), 2, 100)
    weights1 = 1 / (np.exp(20 * (dist_sq_proxy - 0.5)) + 1)
    weights2 = np.power(2 * dist_sq_proxy, -(64 + 2))
    import matplotlib.pyplot as plt
    plt.plot(dist_sq_proxy, weights1, label='Fermi')
    plt.plot(dist_sq_proxy, weights2, label='Riesz')
    plt.legend()
    plt.show()