import sys
import os
import torch

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimize import KissingNumberOptimizer
from src.visualize import plot_2d_solution, plot_loss_history
from src.utils import check_overlaps

def main():
    # 2D Kissing Number is known to be 6
    N = 6
    DIM = 2
    
    print(f"Running optimization for N={N}, d={DIM}...")
    
    optimizer = KissingNumberOptimizer(n_spheres=N, dim=DIM, lr=0.05)
    final_points, history = optimizer.optimize(iterations=2000)
    
    # Check results
    stats = check_overlaps(final_points)
    print("\nOptimization Results:")
    print(f"Minimum pairwise distance: {stats['min_distance']:.4f}")
    print(f"Number of overlaps: {stats['num_overlaps']}")
    print(f"Valid configuration: {stats['is_valid']}")
    
    # Visualize
    # Note: In a headless environment, plt.show() might not work. 
    # You might want to save the figure instead.
    try:
        plot_loss_history(history)
        plot_2d_solution(final_points.cpu().numpy())
    except Exception as e:
        print(f"Visualization failed (likely no display): {e}")

if __name__ == "__main__":
    main()
