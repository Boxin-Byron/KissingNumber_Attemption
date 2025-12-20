import sys
import os
import torch
import argparse
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimize import KissingNumberOptimizer
from src.utils import check_overlaps, check_overlaps_batched

def run_batched_optimization(n, dim, batch_size=10, max_steps=100000, lr=0.05, device='cuda'):
    """
    Runs a batched optimization attempt.
    Returns the points, validity status, and min distances for the batch.
    """
    optimizer = KissingNumberOptimizer(n_spheres=n, dim=dim, batch_size=batch_size, lr=lr, device=device)
    
    # --- Phase 0: Global Repulsion Warmup (Riesz Energy Annealing) ---
    # We anneal 's' from 2.0 (Long range, global structure) to 20.0 (Short range, hard repulsion).
    # This guides points from a uniform distribution to a separated packing.
    riesz_schedule = [2.0, 5.0, 10.0, 20.0]
    steps_per_riesz = 400 // len(riesz_schedule)
    
    print(f"  [Warmup] Running Riesz Energy Annealing s={riesz_schedule}...")
    
    # Use a slightly larger LR for warmup to allow movement
    warmup_optimizer = torch.optim.Adam([optimizer.points], lr=0.02)
    optimizer.optimizer = warmup_optimizer
    
    for s_val in riesz_schedule:
        for i in range(steps_per_riesz):
            optimizer.step_riesz(s=s_val)
        
    # Check stats after warmup
    stats_warmup = check_overlaps_batched(optimizer.points)
    valid_min_dists_warmup = stats_warmup['min_distance'][~torch.isnan(stats_warmup['min_distance'])]
    if valid_min_dists_warmup.numel() > 0:
        print(f"  [Warmup] Finished. Avg Min Dist: {valid_min_dists_warmup.mean().item():.4f} (Max: {valid_min_dists_warmup.max().item():.4f})")
    
    # Re-initialize main optimizer for the Hard Sphere phase
    optimizer.optimizer = torch.optim.Adam([optimizer.points], lr=lr)
    
    # --- Phase 1: Hard Sphere Optimization with Target Expansion ---
    # CRITICAL TRICK: We set the target min_dist slightly higher than 2.0 (e.g., 2.05).
    # Why? In standard loss, pairs with d=2.01 have 0 gradient and act as "lazy obstacles".
    # By targeting 2.05, even "valid" spheres are pushed apart, creating a global pressure
    # that helps squeeze the difficult pairs (d=1.95) into place.
    # Re-initialize main optimizer for the Hard Sphere phase
    optimizer.optimizer = torch.optim.Adam([optimizer.points], lr=lr)
    
    # Use a scheduler to decay learning rate for fine-tuning
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer.optimizer, mode='min', factor=0.5, patience=200
    )
    
    best_loss = float('inf')
    
    # --- Phase 1: Hard Sphere Optimization ---
    # We use pure Hard Sphere loss here.
    # Why? Because adding LJ attraction (alpha=0.1) in Phase 1 proved to be detrimental.
    # It forces spheres into local minima too early, preventing the "sliding" motion needed to find the global optimum.
    # Pure Hard Sphere loss allows spheres to "drift" along the zero-gradient plateau (r>2) until they hit another constraint,
    # which effectively explores the configuration space better than being trapped in a soft LJ well.
    
    # Revert to fixed target 2.0.
    # Dynamic target (1.8 -> 2.0) failed because it made the optimization too easy in the beginning,
    # allowing spheres to settle into low-quality local minima that satisfied d>1.8 but were topologically wrong for d>2.0.
    # When the target increased, these wrong topologies couldn't be fixed.
    # Hard constraint d>2.0 from the start forces the system to reject bad topologies immediately.
    
    optimizer.min_dist = 2.0
    
    for i in range(max_steps):
        loss = optimizer.step()
        scheduler.step(loss)
        
        if loss < best_loss:
            best_loss = loss
            
        # Early stopping if loss is extremely small
        if loss < 1e-7:
            break
            
    stats = check_overlaps_batched(optimizer.points)
    
    # Check for immediate success
    if stats['is_valid'].any():
        print(f"  [Optimization] Found {stats['is_valid'].sum().item()} valid configuration(s). Skipping fine-tuning.")
        return optimizer.points, stats['is_valid'], stats['min_distance']
    
    # Fine-tuning phase: If any in batch are close (min_dist > 1.90)
    needs_finetune = stats['min_distance'] > 1.95
    
    if needs_finetune.any():
        count = needs_finetune.sum().item()
        
        # Get final learning rate from training phase
        final_train_lr = optimizer.optimizer.param_groups[0]['lr']
        
        print(f"  [Fine-tuning] {count} candidates > 1.95. Switching to Centered LJ (Target=2.0)...")
        print(f"  [Fine-tuning] Inheriting learning rate: {final_train_lr:.2e}")
        
        # Calculate initial min distances for comparison
        initial_min_dists = stats['min_distance'][needs_finetune]
        
        # Re-initialize optimizer with the final learning rate
        # Ensure a minimum LR for fine-tuning, as the previous scheduler might have decayed it too much
        finetune_lr = max(final_train_lr * 256, 5e-4)
        print(f"  [Fine-tuning] Setting LR to: {finetune_lr:.2e}")
        
        optimizer.optimizer = torch.optim.Adam([optimizer.points], lr=finetune_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer.optimizer, mode='min', factor=0.8, patience=50
    )
        
        fine_tune_steps = 4000
        
        # Annealing schedule: Start with moderate hardness to prevent crushing, then harden.
        # p=20: Stiff enough to prevent collapse, but soft enough to allow compaction.
        # p=64: Extremely hard, approaching true hard sphere.
        p_schedule = [20, 24, 32, 40, 48, 64, 80, 100, 128]
        steps_per_stage = fine_tune_steps // len(p_schedule)
        
        print(f"  [Fine-tuning] Starting Annealing Schedule: p={p_schedule}")
        print(f"  [Fine-tuning] Using alpha=0.4 (60% HardSphere + 40% LJ Attraction for Compaction)")

        for p_val in p_schedule:
            # Reset scheduler for each stage to allow exploration
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, mode='min', factor=0.8, patience=20)
            
            for i in range(steps_per_stage):
                # Increased alpha to 0.4 to allow LJ attraction to pull spheres together and fix jamming
                loss = optimizer.step_centered_lj(p=p_val, alpha=0.4)
                scheduler.step(loss)
            
            # Optional: Print progress between stages
            stats = check_overlaps_batched(optimizer.points)
            valid_min_dists = stats['min_distance'][~torch.isnan(stats['min_distance'])]
            if valid_min_dists.numel() > 0:
               print(f"    p={p_val} done. Best Dist: {valid_min_dists.max().item():.4f}")

        # Re-check stats after fine-tuning
        stats = check_overlaps_batched(optimizer.points)
        
        # Handle NaNs in output for printing
        valid_min_dists = stats['min_distance'][~torch.isnan(stats['min_distance'])]
        
        # Calculate improvement ratio
        final_min_dists = stats['min_distance'][needs_finetune]
        # Filter out NaNs for ratio calculation
        valid_mask = ~torch.isnan(final_min_dists)
        if valid_mask.any():
            ratios = final_min_dists[valid_mask] / initial_min_dists[valid_mask]
            avg_ratio = ratios.mean().item()
            print(f"  [Fine-tuning] Avg Improvement Ratio (Final/Initial): {avg_ratio:.4f}")
        
        if valid_min_dists.numel() > 0:
            print(f"  [Fine-tuning] Finished. Best Min Dist in batch: {valid_min_dists.max().item():.6f}")
        else:
            print(f"  [Fine-tuning] Finished. All results are NaN.")

    return optimizer.points, stats['is_valid'], stats['min_distance']

def find_max_n(dim, start_n, max_retries=10):
    """
    Iteratively searches for the maximum N for a given dimension.
    Uses batching to run 'max_retries' attempts in parallel.
    """
    current_n = start_n
    best_n_found = 0
    
    print(f"Starting search for Dimension {dim}, starting at N={start_n}")
    print(f"Batch Size (Parallel Attempts): {max_retries}")
    print("-" * 50)
    
    while True:
        print(f"\nTesting N = {current_n}...")
        
        start_time = time.time()
        points, is_valid, min_dists = run_batched_optimization(current_n, dim, batch_size=max_retries)
        duration = time.time() - start_time
        
        if is_valid.any():
            # Find the best valid configuration
            valid_indices = torch.nonzero(is_valid).squeeze()
            if valid_indices.dim() == 0:
                best_idx = valid_indices.item()
            else:
                best_idx = valid_indices[0].item()
                
            best_p = points[best_idx]
            best_min_dist = min_dists[best_idx].item()
            
            print(f" SUCCESS! (Min Dist: {best_min_dist:.6f}, Time: {duration:.2f}s)")
            best_n_found = current_n
            
            # Save successful configuration
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "outputs", f"{dim}d")
            os.makedirs(output_dir, exist_ok=True)
            
            save_path = os.path.join(output_dir, f"results_d{dim}_n{current_n}.pt")
            torch.save(best_p, save_path)
            print(f"  Saved to: {save_path}")
            
            print(f"N={current_n} is possible. Moving to N={current_n+1}")
            current_n += 1
        else:
            print(f" Failed. Best Min Dist: {min_dists.max().item():.6f}")
            print(f"\nCould not find valid configuration for N={current_n} after {max_retries} parallel attempts.")
            print(f"Stopping search.")
            print(f"Maximum N found for D={dim} is: {best_n_found}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find Max Kissing Number for a dimension')
    parser.add_argument('--dim', type=int, required=True, help='Dimension to search')
    parser.add_argument('--start_n', type=int, required=True, help='Starting number of spheres')
    parser.add_argument('--retries', type=int, default=1000, help='Number of retries per N (Batch Size)')
    
    args = parser.parse_args()
    
    find_max_n(args.dim, args.start_n, args.retries)
