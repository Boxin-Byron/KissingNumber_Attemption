import torch
import torch.optim as optim
from .loss import kissing_number_loss, centered_lennard_jones_loss
from tqdm import tqdm

class KissingNumberOptimizer:
    def __init__(self, n_spheres, dim, batch_size=1, lr=0.01, min_dist=2.0, device='cpu'):
        self.n_spheres = n_spheres
        self.dim = dim
        self.batch_size = batch_size
        self.lr = lr
        self.min_dist = min_dist
        self.device = device
        
        # Initialize points on the sphere of radius 2
        # We start with random normal and normalize to length 2
        if batch_size > 1:
            self.points = torch.randn(batch_size, n_spheres, dim, device=device)
        else:
            self.points = torch.randn(n_spheres, dim, device=device)
            
        self.points = self._normalize(self.points)
        self.points.requires_grad_(True)
        
        self.optimizer = optim.Adam([self.points], lr=lr)
        
    def _normalize(self, points):
        """Project points onto the sphere of radius 2."""
        return 2.0 * points / torch.norm(points, dim=-1, keepdim=True)

    def step(self):
        self.optimizer.zero_grad()
        
        # Calculate loss
        loss = kissing_number_loss(self.points, self.min_dist)
        
        loss.backward()
        self.optimizer.step()
        
        # Project back to sphere surface after update
        with torch.no_grad():
            self.points.data = self._normalize(self.points.data)
            
        return loss.item()

    def step_centered_lj(self, p=20, alpha = 1e-1):
        """
        Performs a single optimization step using Centered Lennard-Jones loss.
        """
        self.optimizer.zero_grad()
        
        loss = alpha * centered_lennard_jones_loss(self.points, self.min_dist, p=p)
        loss += (1 - alpha) * kissing_number_loss(self.points, self.min_dist)
        
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            self.points.data = self._normalize(self.points.data)
            
        return loss.item()

    def optimize(self, iterations=1000, verbose=True):
        history = []
        iterator = tqdm(range(iterations)) if verbose else range(iterations)
        
        for i in iterator:
            loss_val = self.step()
            history.append(loss_val)
            if verbose and i % 100 == 0:
                iterator.set_description(f"Loss: {loss_val:.6f}")
                
        return self.points.detach(), history
