"""
Experiment: 2D Kissing Number using MIS approach

The exact answer for d=2 is K(2) = 6.
We will test our method with different numbers of candidate points.
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sampling import sample_sphere_uniform
from src.graph import build_conflict_graph, analyze_conflict_graph
from src.mis_solver import solve_mis
from src.visualize import plot_2d_configuration, plot_distance_histogram
from src.refine import refine_repulsion_projected
from src.soft_hard import solve_soft_then_repair


def run_experiment_2d(n_candidates=100, random_state=42, method='auto',
                      sampling_method='uniform', visualize=True, save_output=False,
                      refine_steps=0, refine_step_size=0.02, refine_min_dist=2.0,
                      soft_delta=0.0, repair_steps=0, repair_step_size=0.02,
                      repair_restarts=3, validate_tol=1e-5):
    """
    Run 2D Kissing Number experiment.
    
    Parameters:
    -----------
    n_candidates : int
        Number of candidate points to sample on the circle
    random_state : int
        Random seed for reproducibility
    method : str
        MIS solving method (Gurobi only; kept as auto/gurobi for compatibility)
    sampling_method : str
        'uniform'
        Number of candidate points to sample on the circle
    random_state : int
        Random seed for reproducibility
    method : str
        MIS solving method ('auto', 'gurobi', 'cvxpy', 'greedy', 'clique')
    visualize : bool
        Whether to create visualizations
    save_output : bool
        Whether to save results to file
        
    Returns:
    --------
    results : dict
        Dictionary containing experiment results
    """
    print("="*70)
    print(f"2D Kissing Number Experiment (K(2) = 6)")
    print("="*70)
    print(f"Configuration:")
    print(f"  Candidate points: {n_candidates}")
    print(f"  Sampling method: {sampling_method}")
    print(f"  MIS method: {method}")
    if refine_steps and refine_steps > 0:
        print(f"  Refinement: repulsion_projected (steps={refine_steps}, step_size={refine_step_size}, min_dist={refine_min_dist})")
    if (soft_delta and soft_delta > 0) or (repair_steps and repair_steps > 0):
        print(f"  Soft→Hard: soft_delta={soft_delta}, repair_steps={repair_steps}, repair_restarts={repair_restarts}, validate_tol={validate_tol}")
    print(f"  Random seed: {random_state}")
    print()
    
    # Step 1: Sample candidate points (uniform on radius-2 circle)
    print("Step 1: Sampling candidate points on circle of radius 2...")
    start_time = time.time()
    points = sample_sphere_uniform(n=n_candidates, dim=2, radius=2.0, random_state=random_state)

    # Optional refinement before graph build
    if refine_steps and refine_steps > 0:
        points = refine_repulsion_projected(
            points,
            radius=2.0,
            min_dist=refine_min_dist,
            steps=refine_steps,
            step_size=refine_step_size,
            warmup_steps=max(50, refine_steps // 2),
            riesz_s=2.0,
        )
    
    sampling_time = time.time() - start_time
    print(f"  Sampled {len(points)} points in {sampling_time:.4f}s")
    print()
    
    # Step 2-4: Graph/MIS (optionally soft) + verify (optionally hard-repaired)
    if (soft_delta and soft_delta > 0) or (repair_steps and repair_steps > 0):
        print("Step 2-4: Soft MIS then hard repair...")
        start_time = time.time()

        result = solve_soft_then_repair(
            points=points,
            hard_min_dist=2.0,
            soft_delta=float(soft_delta),
            graph_epsilon=1e-4,
            mis_method=method,
            time_limit=300,
            repair_steps=int(repair_steps),
            repair_step_size=float(repair_step_size),
            repair_restarts=int(repair_restarts),
            validate_tol=float(validate_tol),
            rng=np.random.default_rng(random_state),
            verbose=True,
        )
        graph_time = 0.0  # accounted inside helper prints; keep fields for compatibility
        solve_time = time.time() - start_time

        mis_nodes = result.soft_mis_nodes
        mis_size = result.soft_mis_size
        method_used = f"{method}+soft_hard"
        stats = {
            'n_nodes': len(points),
            'n_edges': None,
            'density': None,
            'avg_degree': None,
        }

        # Verify on repaired points if available, otherwise on soft-selected points.
        if result.repaired_points is not None:
            selected_points = result.repaired_points
        else:
            selected_points = points[np.asarray(mis_nodes, dtype=int)]

        print(f"\n  MIS size (soft seed): {mis_size}")
        print(f"  Method used: {method_used}")
        print(f"  Pipeline time: {solve_time:.4f}s")
        print()
        print("Step 4: Verifying (hard constraint)...")
    else:
        # Step 2: Build conflict graph
        print("Step 2: Building conflict graph...")
        start_time = time.time()
        G = build_conflict_graph(points, min_dist=2.0, epsilon=1e-4)
        graph_time = time.time() - start_time
        print(f"  Built graph in {graph_time:.4f}s")

        # Analyze graph
        stats = analyze_conflict_graph(G)
        print(f"\nGraph statistics:")
        print(f"  Nodes: {stats['n_nodes']}")
        print(f"  Edges: {stats['n_edges']}")
        print(f"  Density: {stats['density']:.4f}")
        print(f"  Average degree: {stats.get('avg_degree', 0):.2f}")
        print()

        # Step 3: Solve MIS
        print("Step 3: Solving Maximum Independent Set...")
        mis_nodes, mis_size, method_used, solve_time = solve_mis(
            G, method=method, time_limit=300, verbose=True
        )
        print(f"\n  MIS size: {mis_size}")
        print(f"  Method used: {method_used}")
        print(f"  Solve time: {solve_time:.4f}s")
        print()

        # Step 4: Verify solution
        print("Step 4: Verifying solution...")
        selected_points = points[mis_nodes]
    
    # Check pairwise distances
    from scipy.spatial.distance import pdist
    distances = pdist(selected_points)
    min_distance = np.min(distances)
    
    print(f"  Minimum pairwise distance: {min_distance:.6f}")
    print(f"  Required minimum: 2.0")
    
    if min_distance >= 2.0 - float(validate_tol):
        print(f"  ✓ Solution is VALID (no overlaps)")
    else:
        print(f"  ✗ Solution has overlaps!")
    
    # Compare to known answer
    known_answer = 6
    print(f"\n  Known K(2) = {known_answer}")
    print(f"  Our result: {mis_size}")
    
    if mis_size == known_answer:
        print(f"  ✓ EXACT match!")
    elif mis_size < known_answer:
        print(f"  Lower bound: {mis_size}/{known_answer}")
    else:
        print(f"  Warning: Result exceeds known answer (possible error)")
    print()
    
    # Step 5: Visualization
    if visualize:
        print("Step 5: Creating visualizations...")
        
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'outputs', '2d')
        os.makedirs(output_dir, exist_ok=True)
        
        # In soft→hard mode, the final configuration is `selected_points` (may be repaired),
        # and `mis_nodes` no longer indexes the right point set. Plot the final configuration.
        if (soft_delta and soft_delta > 0) or (repair_steps and repair_steps > 0):
            tmp_points = selected_points
            tmp_nodes = list(range(tmp_points.shape[0]))
        else:
            tmp_points = points
            tmp_nodes = mis_nodes

        # Configuration plot
        save_path = os.path.join(output_dir, f'config_n{n_candidates}.png')
        plot_2d_configuration(
            tmp_points,
            tmp_nodes,
            title=f"2D Kissing Number (n={mis_size}, candidates={n_candidates})",
            save_path=save_path,
            show=False,
        )
        
        # Distance histogram
        save_path = os.path.join(output_dir, f'distances_n{n_candidates}.png')
        plot_distance_histogram(
            tmp_points,
            tmp_nodes,
            min_dist=2.0,
            title=f"Pairwise Distances (final n={mis_size})",
            save_path=save_path,
            show=False,
        )
        print()
    
    # Prepare results
    results = {
        'dimension': 2,
        'n_candidates': n_candidates,
        'sampling_method': sampling_method,
        'refine_steps': refine_steps,
        'refine_step_size': refine_step_size,
        'refine_min_dist': refine_min_dist,
        'method': method_used,
        'mis_size': mis_size,
        'mis_nodes': mis_nodes,
        'points': points,
        'min_distance': min_distance,
        'sampling_time': sampling_time,
        'graph_time': graph_time,
        'solve_time': solve_time,
        'total_time': sampling_time + graph_time + solve_time,
        'graph_stats': stats,
        'is_valid': min_distance >= 2.0 - float(validate_tol),
        'matches_known': mis_size == known_answer
    }
    
    # Save results
    if save_output:
        import pickle
        output_file = os.path.join(output_dir, f'results_d2_n{n_candidates}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results to {output_file}")
    
    return results


def run_multiple_experiments():
    """
    Run experiments with different numbers of candidates.
    """
    print("\n" + "="*70)
    print("Running multiple experiments with different candidate counts")
    print("="*70 + "\n")
    
    candidate_counts = [20, 50, 100, 200, 500]
    all_results = []
    
    for n_cand in candidate_counts:
        print(f"\n{'='*70}")
        print(f"Testing with {n_cand} candidates")
        print(f"{'='*70}\n")
        
        results = run_experiment_2d(
            n_candidates=n_cand,
            random_state=42,
            method='auto',
            visualize=(n_cand <= 100),  # Only visualize small cases
            save_output=True
        )
        
        all_results.append(results)
        
        print(f"\nSummary for n={n_cand}:")
        print(f"  MIS size: {results['mis_size']}")
        print(f"  Method: {results['method']}")
        print(f"  Total time: {results['total_time']:.4f}s")
        print(f"  Valid: {results['is_valid']}")
        print()
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Candidates':<12} {'MIS Size':<10} {'Method':<15} {'Time (s)':<10} {'Valid':<8}")
    print("-"*70)
    
    for res in all_results:
        print(f"{res['n_candidates']:<12} {res['mis_size']:<10} {res['method']:<15} "
              f"{res['total_time']:<10.4f} {str(res['is_valid']):<8}")
    
    print("="*70)
    print(f"\nKnown answer: K(2) = 6")
    print(f"Best result: {max(r['mis_size'] for r in all_results)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='2D Kissing Number Experiment')
    parser.add_argument('-n', '--candidates', type=int, default=1000,
                       help='Number of candidate points (default: 1000)')
    parser.add_argument('-s', '--sampling', type=str, default='uniform',
                       choices=['uniform'],
                       help='Sampling method (default: uniform)')
    parser.add_argument('-m', '--method', type=str, default='auto',
                       choices=['auto', 'gurobi'],
                       help='MIS solving method (Gurobi only; kept as auto/gurobi for compatibility)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    parser.add_argument('--multiple', action='store_true',
                       help='Run multiple experiments with different candidate counts')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    parser.add_argument('--restarts', type=int, default=1,
                       help='Number of random restarts (only meaningful for uniform sampling). Best result is kept. (default: 1)')
    parser.add_argument('--refine-steps', type=int, default=0,
                       help='Projected refinement steps before building the graph (default: 0)')
    parser.add_argument('--refine-step-size', type=float, default=0.02,
                       help='Step size for refinement (default: 0.02)')
    parser.add_argument('--refine-min-dist', type=float, default=2.0,
                       help='Target min_dist for refinement (default: 2.0)')

    parser.add_argument('--soft-delta', type=float, default=0.02,
                       help='Soft relaxation: soft_min = 2*(1-soft_delta). (default: 0.02)')
    parser.add_argument('--repair-steps', type=int, default=3,
                       help='Hard repair steps after soft MIS (0 disables). (default: 3)')
    parser.add_argument('--repair-step-size', type=float, default=0.02,
                       help='Step size for hard repair. (default: 0.02)')
    parser.add_argument('--repair-restarts', type=int, default=3,
                       help='Number of repair retries with tiny jitter. (default: 3)')
    parser.add_argument('--validate-tol', type=float, default=1e-5,
                       help='Accept if min_dist >= 2-validate_tol. (default: 1e-5)')
    
    args = parser.parse_args()
    
    if args.multiple:
        run_multiple_experiments()
    else:
        # Multi-restart: keep the best MIS size (validity checked inside).
        best = None
        for k in range(max(1, args.restarts)):
            seed_k = args.seed + k
            res = run_experiment_2d(
                n_candidates=args.candidates,
                random_state=seed_k,
                method=args.method,
                sampling_method=args.sampling,
                visualize=(not args.no_viz) and (k == 0),
                save_output=args.save,
                refine_steps=args.refine_steps,
                refine_step_size=args.refine_step_size,
                refine_min_dist=args.refine_min_dist,
                soft_delta=args.soft_delta,
                repair_steps=args.repair_steps,
                repair_step_size=args.repair_step_size,
                repair_restarts=args.repair_restarts,
                validate_tol=args.validate_tol,
            )
            if (best is None) or (res['mis_size'] > best['mis_size']):
                best = res

        results = best
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
