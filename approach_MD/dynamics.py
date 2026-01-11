import numpy as np
from geometry import *


class SphericalOptimizer:
    def __init__(self, n_particles, n_dim, dt=0.01, damping=0.9, mass=1.0):
        self.dt = dt
        self.damping = damping # 0.0 ~ 1.0, 越小阻尼越大
        self.mass = mass
        self.v = np.zeros((n_particles, n_dim))
        
    def step(self, X, potential):
        """
        执行一步动力学更新
        X: 当前位置 (N, n)
        potential: 势函数对象
        """
        # 1. 将力投影到切空间 (Riemannian Gradient)
        f_tan = project_forces(X, potential.compute_force(X))
        
        # 2. 更新速度 (带阻尼)
        # a = F / m - gamma * v
        # 这里用简化的动量更新： v = v * damping + f * dt
        self.v = self.v * self.damping + (f_tan / self.mass) * self.dt
        
        # 3. 更新位置
        X_new = X + self.v * self.dt
        
        # 4. 约束投影 (Retraction)
        X_new = normalize(X_new)
        
        return X_new
        
    def reset_velocity(self):
        self.v.fill(0.0)