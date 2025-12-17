import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimize import KissingNumberOptimizer
from src.visualize import plot_loss_history
from src.utils import check_overlaps

def plot_3d_solution(points, title="3D Kissing Number Solution"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot central sphere (wireframe or just a point)
    ax.scatter([0], [0], [0], color='gray', s=100, label='Center', alpha=0.5)
    
    # Plot points
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    
    ax.scatter(xs, ys, zs, c='b', s=200, depthshade=True, label='Spheres')
    
    for i in range(len(points)):
        ax.text(xs[i], ys[i], zs[i], str(i+1), color='black')
        # Draw line to center
        ax.plot([0, xs[i]], [0, ys[i]], [0, zs[i]], 'k--', alpha=0.2)
        
    # Draw a wireframe sphere for reference (radius 2)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = 2 * np.cos(u) * np.sin(v)
    y = 2 * np.sin(u) * np.sin(v)
    z = 2 * np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)

    ax.set_title(title)
    ax.legend()
    plt.show()

def main():
    # 3D Kissing Number is known to be 12
    N = 12
    DIM = 3
    
    print(f"Running optimization for N={N}, d={DIM}...")
    
    optimizer = KissingNumberOptimizer(n_spheres=N, dim=DIM, lr=0.05)
    final_points, history = optimizer.optimize(iterations=5000) # More iterations for 3D
    
    # Check results
    stats = check_overlaps(final_points)
    print("\nOptimization Results:")
    print(f"Minimum pairwise distance: {stats['min_distance']:.4f}")
    print(f"Number of overlaps: {stats['num_overlaps']}")
    print(f"Valid configuration: {stats['is_valid']}")
    
    try:
        plot_loss_history(history)
        plot_3d_solution(final_points.cpu().numpy())
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()
