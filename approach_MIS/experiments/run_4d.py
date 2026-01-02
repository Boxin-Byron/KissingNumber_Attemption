"""
Experiment: 4D Kissing Number using MIS approach

The exact answer for d=4 is K(4) = 24.
We will test our method with different numbers of candidate points.
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sampling import sample_sphere_uniform, sample_sphere_fps
from src.graph import build_conflict_graph, analyze_conflict_graph
from src.mis_solver import solve_mis
from src.visualize import plot_distance_histogram
from src.refine import refine_repulsion_projected
from src.soft_hard import solve_soft_then_repair


def run_experiment_4d(n_candidates=5000, random_state=42, method='auto',
                      visualize=True, save_output=False, time_limit=600,
                      refine_steps=0, refine_step_size=0.05, refine_min_dist=2.0,
                      soft_delta=0.0, repair_steps=0, repair_step_size=0.05,
                      repair_restarts=3, validate_tol=1e-5,
                      repair_stage_mins=None, repair_stage_fracs=None,
                      sampling='uniform', oversample=20000, fps_start='random'):
    """
    Run 4D Kissing Number experiment.
    
    Parameters:
    -----------
    n_candidates : int
        Number of candidate points to sample on the 3-sphere
    random_state : int
        Random seed for reproducibility
    method : str
        MIS solving method (Gurobi only; kept as auto/gurobi for compatibility)
    visualize : bool
        Whether to create visualizations (histogram only for 4D)
    save_output : bool
        Whether to save results to file
    time_limit : float
        Time limit for MIS solver (seconds)
        
    Returns:
    --------
    results : dict
        Dictionary containing experiment results
    """
    print("="*70)
    print(f"4D Kissing Number Experiment (K(4) = 24)")
    print("="*70)
    print(f"Configuration:")
    print(f"  Candidate points: {n_candidates}")
    if sampling and sampling != 'uniform':
        print(f"  Sampling: {sampling} (oversample={oversample}, fps_start={fps_start})")
    print(f"  MIS method: {method}")
    print(f"  Time limit: {time_limit}s")
    if refine_steps and refine_steps > 0:
        print(f"  Refinement: repulsion_projected (steps={refine_steps}, step_size={refine_step_size}, min_dist={refine_min_dist})")
    if (soft_delta and soft_delta > 0) or (repair_steps and repair_steps > 0):
        print(f"  Soft→Hard: soft_delta={soft_delta}, repair_steps={repair_steps}, repair_restarts={repair_restarts}, validate_tol={validate_tol}")
        if repair_stage_mins is not None:
            print(f"           repair_stage_mins={repair_stage_mins}, repair_stage_fracs={repair_stage_fracs}")
    print(f"  Random seed: {random_state}")
    print()
    
    # Step 1: Sample candidate points
    print("Step 1: Sampling candidate points on 3-sphere (in 4D space) of radius 2...")
    start_time = time.time()

    if sampling == 'fps':
        points = sample_sphere_fps(
            n_keep=n_candidates,
            dim=4,
            radius=2.0,
            oversample=int(oversample),
            start=str(fps_start),
            random_state=random_state,
        )
    else:
        points = sample_sphere_uniform(n=n_candidates, dim=4, radius=2.0,
                                      random_state=random_state)
    sampling_time = time.time() - start_time
    print(f"  Sampled {len(points)} points in {sampling_time:.4f}s")

    # Optional refinement: push close pairs apart while staying on sphere
    if refine_steps and refine_steps > 0:
        print(f"  Refining candidates with projected repulsion...")
        refine_start = time.time()
        points = refine_repulsion_projected(
            points,
            radius=2.0,
            min_dist=refine_min_dist,
            steps=refine_steps,
            step_size=refine_step_size,
        )
        refine_time = time.time() - refine_start
        print(f"  Refinement done in {refine_time:.4f}s")
    print()
    
    # Step 2-4: Graph/MIS (optionally soft) + verify (optionally hard-repaired)
    if (soft_delta and soft_delta > 0) or (repair_steps and repair_steps > 0):
        print("Step 2-4: Soft MIS then hard repair...")
        start_pipeline = time.time()

        result = solve_soft_then_repair(
            points=points,
            hard_min_dist=2.0,
            soft_delta=float(soft_delta),
            graph_epsilon=1e-6,
            mis_method=method,
            time_limit=float(time_limit),
            repair_steps=int(repair_steps),
            repair_step_size=float(repair_step_size),
            repair_restarts=int(repair_restarts),
            repair_stage_mins=repair_stage_mins,
            repair_stage_fracs=repair_stage_fracs,
            validate_tol=float(validate_tol),
            rng=np.random.default_rng(random_state),
            verbose=True,
        )
        solve_time = time.time() - start_pipeline
        graph_time = 0.0
        stats = {
            'n_nodes': len(points),
            'n_edges': None,
            'density': None,
            'avg_degree': None,
        }

        mis_nodes = result.soft_mis_nodes
        mis_size = result.soft_mis_size
        method_used = f"{method}+soft_hard"

        if result.repaired_points is not None:
            selected_points = result.repaired_points
        else:
            selected_points = points[np.asarray(mis_nodes, dtype=int)]

        print(f"\n  MIS size (soft seed): {mis_size}")
        print(f"  Method used: {method_used}")
        print(f"  Pipeline time: {solve_time:.4f}s")
        print()
    else:
        # Step 2: Build conflict graph
        print("Step 2: Building conflict graph...")
        start_time = time.time()
        G = build_conflict_graph(points, min_dist=2.0, epsilon=1e-6)
        graph_time = time.time() - start_time
        print(f"  Built graph in {graph_time:.4f}s")

        # Analyze graph
        stats = analyze_conflict_graph(G)
        print(f"\nGraph statistics:")
        print(f"  Nodes: {stats['n_nodes']}")
        print(f"  Edges: {stats['n_edges']}")
        print(f"  Density: {stats['density']:.4f}")
        print(f"  Average degree: {stats.get('avg_degree', 0):.2f}")

        # Estimate solve difficulty
        print(f"\n  Graph complexity assessment:")
        if stats['n_edges'] < 10000:
            print(f"    Small graph - should solve quickly")
        elif stats['n_edges'] < 100000:
            print(f"    Medium graph - may take some time")
        else:
            print(f"    Large graph - will take significant time or may need heuristic")
        print()

        # Step 3: Solve MIS (Gurobi)
        print("Step 3: Solving Maximum Independent Set...")
        print("  (This may take a while for large graphs...)")
        mis_nodes, mis_size, method_used, solve_time = solve_mis(
            G, method=method, time_limit=time_limit, verbose=True
        )
    
    print(f"\n  MIS size: {mis_size}")
    print(f"  Method used: {method_used}")
    print(f"  Solve time: {solve_time:.4f}s")
    print()
    
    # Step 4: Verify solution
    print("Step 4: Verifying solution...")
    if not ((soft_delta and soft_delta > 0) or (repair_steps and repair_steps > 0)):
        selected_points = points[mis_nodes]
    
    # Check pairwise distances
    from scipy.spatial.distance import pdist
    distances = pdist(selected_points)
    min_distance = np.min(distances)
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    
    print(f"  Minimum pairwise distance: {min_distance:.6f}")
    print(f"  Mean pairwise distance: {mean_distance:.6f}")
    print(f"  Maximum pairwise distance: {max_distance:.6f}")
    print(f"  Required minimum: 2.0")
    
    if min_distance >= 2.0 - float(validate_tol):
        print(f"  ✓ Solution is VALID (no overlaps)")
    else:
        print(f"  ✗ Solution has overlaps!")
        violations = np.sum(distances < 2.0 - float(validate_tol))
        print(f"  Number of violations: {violations}")
    
    # Compare to known answer
    known_answer = 24
    print(f"\n  Known K(4) = {known_answer}")
    print(f"  Our result: {mis_size}")
    
    if mis_size == known_answer:
        print(f"  ✓ EXACT match!")
    elif mis_size < known_answer:
        gap = known_answer - mis_size
        print(f"  Lower bound: {mis_size}/{known_answer} (gap: {gap})")
        print(f"  Coverage: {100 * mis_size / known_answer:.1f}%")
    else:
        print(f"  Warning: Result exceeds known answer (possible error)")
    print()
    
    # Step 5: Analysis and Visualization
    if visualize:
        print("Step 5: Creating visualizations...")
        
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'outputs', '4d')
        os.makedirs(output_dir, exist_ok=True)
        
        # Distance histogram (only practical visualization for 4D)
        # IMPORTANT: if we ran soft→hard repair, `mis_nodes` index into the original
        # candidate array, but the final configuration is `selected_points`.
        # So we plot the final configuration if available.
        tag = f"n{n_candidates}_seed{random_state}_mis{mis_size}"
        if (soft_delta and soft_delta > 0) or (repair_steps and repair_steps > 0):
            tmp_points = selected_points
            tmp_nodes = list(range(tmp_points.shape[0]))
            save_path = os.path.join(output_dir, f"distances_final_{tag}.png")
            plot_distance_histogram(
                tmp_points,
                tmp_nodes,
                min_dist=2.0,
                title=f"4D Pairwise Distances (final, n={tmp_points.shape[0]})",
                save_path=save_path,
                show=False,
            )
        else:
            save_path = os.path.join(output_dir, f"distances_{tag}.png")
            plot_distance_histogram(
                points,
                mis_nodes,
                min_dist=2.0,
                title=f"4D Pairwise Distances (MIS={mis_size})",
                save_path=save_path,
                show=False,
            )
        
        # Create a text summary file
        summary_file = os.path.join(output_dir, f'summary_n{n_candidates}.txt')
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("4D Kissing Number Experiment Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Candidates: {n_candidates}\n")
            f.write(f"  Method: {method_used}\n")
            f.write(f"  Time limit: {time_limit}s\n\n")
            f.write(f"Results:\n")
            f.write(f"  MIS size: {mis_size}\n")
            f.write(f"  Known K(4): {known_answer}\n")
            f.write(f"  Gap: {known_answer - mis_size}\n")
            f.write(f"  Coverage: {100 * mis_size / known_answer:.1f}%\n\n")
            f.write(f"Distance statistics:\n")
            f.write(f"  Min: {min_distance:.6f}\n")
            f.write(f"  Mean: {mean_distance:.6f}\n")
            f.write(f"  Max: {max_distance:.6f}\n\n")
            f.write(f"Timing:\n")
            f.write(f"  Sampling: {sampling_time:.4f}s\n")
            f.write(f"  Graph building: {graph_time:.4f}s\n")
            f.write(f"  MIS solving: {solve_time:.4f}s\n")
            f.write(f"  Total: {sampling_time + graph_time + solve_time:.4f}s\n\n")
            f.write(f"Graph statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"  Saved summary to {summary_file}")
        print()
    
    # Prepare results
    results = {
        'dimension': 4,
        'n_candidates': n_candidates,
        'refine_steps': refine_steps,
        'refine_step_size': refine_step_size,
        'refine_min_dist': refine_min_dist,
        'method': method_used,
        'mis_size': mis_size,
        'mis_nodes': mis_nodes,
        'points': points,
        'min_distance': min_distance,
        'mean_distance': mean_distance,
        'max_distance': max_distance,
        'sampling_time': sampling_time,
        'graph_time': graph_time,
        'solve_time': solve_time,
        'total_time': sampling_time + graph_time + solve_time,
        'graph_stats': stats,
        'is_valid': min_distance >= 2.0 - float(validate_tol),
        'matches_known': mis_size == known_answer,
        'gap': known_answer - mis_size,
        'coverage': 100 * mis_size / known_answer
    }
    
    # Save results
    if save_output:
        import pickle
        output_file = os.path.join(output_dir, f'results_d4_n{n_candidates}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results to {output_file}")
    
    return results


def run_multiple_experiments():
    """
    Run experiments with different numbers of candidates.
    """
    print("\n" + "="*70)
    print("Running multiple 4D experiments with different candidate counts")
    print("="*70 + "\n")
    
    # Test with increasing numbers of candidates
    candidate_counts = [500, 1000, 2000, 5000, 10000]
    all_results = []
    
    for n_cand in candidate_counts:
        print(f"\n{'='*70}")
        print(f"Testing with {n_cand} candidates")
        print(f"{'='*70}\n")
        
        results = run_experiment_4d(
            n_candidates=n_cand,
            random_state=42,
            method='auto',
            visualize=True,
            save_output=True,
            time_limit=600
        )
        
        all_results.append(results)
        
        print(f"\nSummary for n={n_cand}:")
        print(f"  MIS size: {results['mis_size']}/24")
        print(f"  Gap: {results['gap']}")
        print(f"  Method: {results['method']}")
        print(f"  Total time: {results['total_time']:.4f}s")
        print(f"  Valid: {results['is_valid']}")
        print()
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Candidates':<12} {'MIS Size':<10} {'Gap':<6} {'Coverage':<10} "
          f"{'Method':<15} {'Time (s)':<10} {'Valid':<8}")
    print("-"*70)
    
    for res in all_results:
        print(f"{res['n_candidates']:<12} {res['mis_size']:<10} {res['gap']:<6} "
              f"{res['coverage']:<10.1f} {res['method']:<15} "
              f"{res['total_time']:<10.2f} {str(res['is_valid']):<8}")
    
    print("="*70)
    print(f"\nKnown answer: K(4) = 24")
    print(f"Best result: {max(r['mis_size'] for r in all_results)}")
    
    # Find best configuration
    best = max(all_results, key=lambda x: x['mis_size'])
    print(f"\nBest configuration: {best['mis_size']} spheres "
          f"with {best['n_candidates']} candidates")
    
    if best['mis_size'] == 24:
        print("🎉 Successfully found optimal solution!")
    else:
        print(f"Recommendation: Try more candidates (>{max(candidate_counts)}) "
              f"or use exact solver (Gurobi)")


if __name__ == "__main__":
    import argparse

    def _parse_float_list(s):
        if s is None:
            return None
        s = str(s).strip()
        if s == "":
            return None
        return [float(x) for x in s.split(",") if str(x).strip() != ""]
    
    parser = argparse.ArgumentParser(description='4D Kissing Number Experiment')
    parser.add_argument('-n', '--candidates', type=int, default=5000,
                       help='Number of candidate points (default: 5000)')
    parser.add_argument('-m', '--method', type=str, default='auto',
                       choices=['auto', 'gurobi'],
                       help='MIS solving method (Gurobi only; kept as auto/gurobi for compatibility)')
    parser.add_argument('-t', '--time-limit', type=int, default=600,
                       help='Time limit for solver in seconds (default: 600)')
    # Visualization / saving are always enabled (outputs go to MIS/outputs/4d).
    parser.add_argument('--multiple', action='store_true',
                       help='Run multiple experiments with different candidate counts')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    parser.add_argument('--sampling', type=str, default='uniform',
                       choices=['uniform', 'fps'],
                       help='Candidate sampling method (default: uniform). fps = oversample then farthest-point-sampling keep-N.')
    parser.add_argument('--oversample', type=int, default=20000,
                       help='When sampling=fps: number of uniform candidates generated before FPS (default: 20000)')
    parser.add_argument('--fps-start', type=str, default='random',
                       choices=['random', '0'],
                       help="When sampling=fps: FPS start point ('random' or '0') (default: random)")

    parser.add_argument('--refine-steps', type=int, default=0,
                       help='Projected repulsion refinement steps applied to candidates before building the graph (default: 0 = disabled)')
    parser.add_argument('--refine-step-size', type=float, default=0.05,
                       help='Step size for refinement (default: 0.05)')
    parser.add_argument('--refine-min-dist', type=float, default=2.0,
                       help='Target min_dist for refinement (default: 2.0). Keep >=2.0 to maintain a hard-sphere bias.')

    parser.add_argument('--soft-delta', type=float, default=0.0,
                       help='Soft relaxation: soft_min = 2*(1-soft_delta). (default: 0.0)')
    parser.add_argument('--repair-steps', type=int, default=0,
                       help='Hard repair steps after soft MIS (0 disables). (default: 0)')
    parser.add_argument('--repair-step-size', type=float, default=0.05,
                       help='Step size for hard repair. (default: 0.05)')
    parser.add_argument('--repair-restarts', type=int, default=3,
                       help='Number of repair retries with tiny jitter. (default: 3)')
    parser.add_argument('--validate-tol', type=float, default=1e-5,
                       help='Accept if min_dist >= 2-validate_tol. (default: 1e-5)')

    parser.add_argument('--repair-stage-mins', type=str, default=None,
                       help='Optional multi-stage repair schedule, comma-separated. Example: "1.90,1.97,2.0". (default: disabled)')
    parser.add_argument('--repair-stage-fracs', type=str, default=None,
                       help='Optional per-stage step fractions, comma-separated. Example: "0.5,0.3,0.2". (default: even split)')
    
    args = parser.parse_args()
    
    method = args.method
    
    if args.multiple:
        run_multiple_experiments()
    else:
        repair_stage_mins = _parse_float_list(args.repair_stage_mins)
        repair_stage_fracs = _parse_float_list(args.repair_stage_fracs)
        results = run_experiment_4d(
            n_candidates=args.candidates,
            random_state=args.seed,
            method=method,
            visualize=True,
            save_output=True,
            time_limit=args.time_limit,
            refine_steps=args.refine_steps,
            refine_step_size=args.refine_step_size,
            refine_min_dist=args.refine_min_dist,
            soft_delta=args.soft_delta,
            repair_steps=args.repair_steps,
            repair_step_size=args.repair_step_size,
            repair_restarts=args.repair_restarts,
            validate_tol=args.validate_tol,
            repair_stage_mins=repair_stage_mins,
            repair_stage_fracs=repair_stage_fracs,
            sampling=args.sampling,
            oversample=args.oversample,
            fps_start=args.fps_start,
        )
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
        print(f"\nFinal result: {results['mis_size']}/24 spheres")
        print(f"Coverage: {results['coverage']:.1f}%")
