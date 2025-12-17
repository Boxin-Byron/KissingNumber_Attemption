import sys
import os
import torch
import argparse
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimize import KissingNumberOptimizer
from src.utils import check_overlaps

def run_optimization_attempt(n, dim, max_steps=5000, lr=0.05, device='cpu'):
    """
    Runs a single optimization attempt for a fixed N.
    Returns the final points and validity status.
    """
    optimizer = KissingNumberOptimizer(n_spheres=n, dim=dim, lr=lr, device=device)
    
    # Use a scheduler to decay learning rate for fine-tuning
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer.optimizer, mode='min', factor=0.5, patience=200
    )
    
    best_loss = float('inf')
    
    for i in range(max_steps):
        loss = optimizer.step()
        scheduler.step(loss)
        
        if loss < best_loss:
            best_loss = loss
            
        # Early stopping if loss is extremely small
        if loss < 1e-7:
            break
            
    stats = check_overlaps(optimizer.points)
    return optimizer.points, stats['is_valid'], stats['min_distance']

def find_max_n(dim, start_n, max_retries=5):
    """
    Iteratively searches for the maximum N for a given dimension.
    """
    current_n = start_n
    best_n_found = 0
    best_config = None
    
    print(f"Starting search for Dimension {dim}, starting at N={start_n}")
    print("-" * 50)
    
    while True:
        print(f"\nTesting N = {current_n}...")
        success = False
        
        for attempt in range(max_retries):
            print(f"  Attempt {attempt+1}/{max_retries}...", end="", flush=True)
            start_time = time.time()
            
            points, is_valid, min_dist = run_optimization_attempt(current_n, dim)
            
            duration = time.time() - start_time
            
            if is_valid:
                print(f" SUCCESS! (Min Dist: {min_dist:.6f}, Time: {duration:.2f}s)")
                success = True
                best_n_found = current_n
                best_config = points
                
                # Save successful configuration
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                output_dir = os.path.join(base_dir, "outputs", f"{dim}d")
                os.makedirs(output_dir, exist_ok=True)
                
                save_path = os.path.join(output_dir, f"results_d{dim}_n{current_n}.pt")
                torch.save(points, save_path)
                print(f"  Saved to: {save_path}")
                break
            else:
                print(f" Failed. (Min Dist: {min_dist:.6f})")
        
        if success:
            print(f"N={current_n} is possible. Moving to N={current_n+1}")
            current_n += 1
        else:
            print(f"\nCould not find valid configuration for N={current_n} after {max_retries} attempts.")
            print(f"Stopping search.")
            print(f"Maximum N found for D={dim} is: {best_n_found}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find Max Kissing Number for a dimension')
    parser.add_argument('--dim', type=int, required=True, help='Dimension to search')
    parser.add_argument('--start_n', type=int, required=True, help='Starting number of spheres')
    parser.add_argument('--retries', type=int, default=10, help='Number of retries per N')
    
    args = parser.parse_args()
    
    find_max_n(args.dim, args.start_n, args.retries)
