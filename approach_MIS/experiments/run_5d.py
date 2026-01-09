"""Experiment: 5D Kissing Number using MIS approach.

Known facts (for reference):
  - Best known lower bound: 40
  - Best known upper bound: 44

This script mirrors the 4D pipeline but in 5D:
  sample candidates on the radius-2 4-sphere in R^5,
  build conflict graph with min_dist (hard or soft),
  solve MIS, and report geometric validation stats.

Focus: get the pipeline running with FPS sampling and soft MIS.
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sampling import sample_sphere_uniform, sample_sphere_fps
from src.graph import build_conflict_graph, analyze_conflict_graph
from src.mis_solver import solve_mis
from src.visualize import plot_distance_histogram, plot_nearest_neighbor_histogram
from src.soft_hard import solve_soft_then_repair


def run_experiment_5d(
    n_candidates=5000,
    random_state=42,
    method='auto',
    time_limit=300,
    soft_delta=0.0,
    visualize=True,
    save_output=True,
    sampling='uniform',
    oversample=20000,
    fps_start='random',
    validate_tol=1e-5,
    repair_steps=0,
    repair_step_size=0.03,
    repair_restarts=10,
    repair_stage_mins=None,
    repair_stage_fracs=None,
):
    print("=" * 70)
    print("5D Kissing Number Experiment")
    print("=" * 70)
    print("Configuration:")
    print(f"  Candidate points: {n_candidates}")
    print(f"  Sampling: {sampling} (oversample={oversample}, fps_start={fps_start})")
    print(f"  MIS method: {method}")
    print(f"  Time limit: {time_limit}s")
    print(f"  soft_delta: {soft_delta}")
    if (soft_delta and soft_delta > 0) or (repair_steps and repair_steps > 0):
        print(
            f"  Soft→Hard: repair_steps={repair_steps}, repair_restarts={repair_restarts}, validate_tol={validate_tol}"
        )
        if repair_stage_mins is not None:
            print(f"           repair_stage_mins={repair_stage_mins}, repair_stage_fracs={repair_stage_fracs}")
    print(f"  Random seed: {random_state}")
    print()

    hard_min = 2.0
    soft_min = hard_min * (1.0 - float(soft_delta))

    # Output dir (for plots + summaries)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', '5d')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Step 1: sample candidates on radius-2 sphere in 5D
    print("Step 1: Sampling candidate points on 4-sphere (in 5D space) of radius 2...")
    t0 = time.time()
    if sampling == 'fps':
        points = sample_sphere_fps(
            n_keep=int(n_candidates),
            dim=5,
            radius=2.0,
            oversample=int(oversample),
            start=str(fps_start),
            random_state=int(random_state),
        )
    else:
        points = sample_sphere_uniform(
            n=int(n_candidates),
            dim=5,
            radius=2.0,
            random_state=int(random_state),
        )
    sampling_time = time.time() - t0
    print(f"  Sampled {len(points)} points in {sampling_time:.4f}s")
    print()

    # Step 2-4: Soft MIS then (optional) hard repair
    if (soft_delta and soft_delta > 0) or (repair_steps and repair_steps > 0):
        print("Step 2-4: Soft MIS then hard repair...")
        t_pipe = time.time()
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
            rng=np.random.default_rng(int(random_state)),
            verbose=True,
        )
        solve_time = time.time() - t_pipe
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
            selected = result.repaired_points
        else:
            selected = points[np.asarray(mis_nodes, dtype=int)]

        print(f"\n  MIS size (soft seed): {mis_size}")
        print(f"  Method used: {method_used}")
        print(f"  Pipeline time: {solve_time:.4f}s")
        print()
        print("Step 4: Verifying distances...")
    else:
        # Step 2: build conflict graph
        print("Step 2: Building conflict graph...")
        t1 = time.time()
        G = build_conflict_graph(points, min_dist=float(soft_min), epsilon=1e-6)
        graph_time = time.time() - t1
        stats = analyze_conflict_graph(G)
        print(f"  Built graph in {graph_time:.4f}s")
        print(f"  Nodes: {stats['n_nodes']}")
        print(f"  Edges: {stats['n_edges']}")
        print(f"  Density: {stats['density']:.4f}")
        print(f"  Average degree: {stats.get('avg_degree', 0):.2f}")
        print()

        # Step 3: solve MIS
        print("Step 3: Solving Maximum Independent Set...")
        mis_nodes, mis_size, method_used, solve_time = solve_mis(
            G, method=method, time_limit=float(time_limit), verbose=True
        )
        print(f"\n  MIS size (soft seed if soft_delta>0): {mis_size}")
        print(f"  Method used: {method_used}")
        print(f"  Solve time: {solve_time:.4f}s")
        print()

        # Step 4: quick verification
        print("Step 4: Verifying distances...")
        selected = points[np.asarray(mis_nodes, dtype=int)]

    # numpy-only min-distance (avoid scipy dependency for this script)
    n = selected.shape[0]
    if n <= 1:
        min_d = float('inf')
    else:
        diff = selected[:, None, :] - selected[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[np.eye(n, dtype=bool)] = np.inf
        min_d = float(np.min(dist))

    print(f"  soft_min_dist required: {soft_min:.6f}")
    print(f"  hard_min_dist required: {hard_min:.6f}")
    print(f"  min pairwise distance (selected): {min_d:.6f}")

    hard_ok = (min_d >= hard_min - float(validate_tol))
    if hard_ok:
        print("  ✓ Hard-valid (passes hard constraint within tolerance)")
    else:
        print("  ✗ Not hard-valid (this is expected for soft runs)")
    print()

    # Step 5: Visualizations (dimension-agnostic)
    if visualize:
        print("Step 5: Saving visualizations (5D)…")
        tag = f"n{int(n_candidates)}_seed{int(random_state)}_soft{float(soft_delta):.3f}_mis{int(mis_size)}"

        # Plot with the selected subset only (histograms).
        # For plot_distance_histogram, we pass `mis_nodes` against the full candidate array.
        # If we repaired, the indices no longer map; in that case, we plot distances using selected_points directly.
        if result is not None and result.repaired_points is not None:
            # Create a synthetic points array where selected points are the whole set.
            tmp_points = selected
            tmp_nodes = list(range(tmp_points.shape[0]))
            dist_path = os.path.join(output_dir, f"distances_repaired_{tag}.png")
            plot_distance_histogram(
                tmp_points,
                tmp_nodes,
                min_dist=2.0,
                title=f"5D pairwise distances (repaired, n={tmp_points.shape[0]})",
                save_path=dist_path,
                show=False,
            )
        else:
            dist_path = os.path.join(output_dir, f"distances_{tag}.png")
            plot_distance_histogram(
                points,
                list(map(int, mis_nodes)),
                min_dist=2.0,
                title=f"5D pairwise distances (MIS={mis_size})",
                save_path=dist_path,
                show=False,
            )

        nn_path = os.path.join(output_dir, f"nearest_neighbor_{tag}.png")
        plot_nearest_neighbor_histogram(
            selected,
            min_dist=2.0,
            title=f"5D nearest-neighbor distances (n={selected.shape[0]})",
            save_path=nn_path,
            show=False,
        )

    # Step 6: Save a small text summary
    if save_output:
        tag = f"n{int(n_candidates)}_seed{int(random_state)}_soft{float(soft_delta):.3f}_mis{int(mis_size)}"
        summary_path = os.path.join(output_dir, f"summary_{tag}.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("5D Kissing Number (MIS approach) summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"n_candidates: {int(n_candidates)}\n")
            f.write(f"sampling: {sampling}\n")
            f.write(f"oversample: {int(oversample)}\n")
            f.write(f"seed: {int(random_state)}\n")
            f.write(f"mis_method: {method}\n")
            f.write(f"time_limit: {float(time_limit)}\n")
            f.write(f"soft_delta: {float(soft_delta)}\n")
            f.write(f"soft_min_dist: {float(soft_min)}\n")
            f.write(f"hard_min_dist: {float(hard_min)}\n")
            f.write(f"mis_size: {int(mis_size)}\n")
            f.write(f"min_pairwise_distance: {float(min_d)}\n")
            f.write(f"hard_valid: {bool(hard_ok)} (tol={float(validate_tol)})\n")
            if (soft_delta and soft_delta > 0) or (repair_steps and repair_steps > 0):
                f.write(f"repair_steps: {int(repair_steps)}\n")
                f.write(f"repair_step_size: {float(repair_step_size)}\n")
                f.write(f"repair_restarts: {int(repair_restarts)}\n")
                f.write(f"repair_stage_mins: {repair_stage_mins}\n")
                f.write(f"repair_stage_fracs: {repair_stage_fracs}\n")
        print(f"Saved summary to {summary_path}")

    return {
        'dimension': 5,
        'n_candidates': int(n_candidates),
        'sampling': sampling,
        'oversample': int(oversample),
        'soft_delta': float(soft_delta),
        'soft_min_dist': float(soft_min),
        'hard_min_dist': float(hard_min),
        'mis_size': int(mis_size),
        'mis_nodes': list(map(int, mis_nodes)),
        'min_distance': float(min_d),
        'is_hard_valid': bool(hard_ok),
        'sampling_time': float(sampling_time),
        'graph_time': float(graph_time),
        'solve_time': float(solve_time),
        'graph_stats': stats,
    }


if __name__ == '__main__':
    import argparse

    def _parse_float_list(s):
        if s is None:
            return None
        s = str(s).strip()
        if s == "":
            return None
        return [float(x) for x in s.split(",") if str(x).strip() != ""]

    parser = argparse.ArgumentParser(description='5D Kissing Number Experiment')
    parser.add_argument('-n', '--candidates', type=int, default=5000,
                        help='Number of candidate points (default: 5000)')
    parser.add_argument('-m', '--method', type=str, default='auto',
                        choices=['auto', 'gurobi'],
                        help='MIS solving method (Gurobi only; kept as auto/gurobi for compatibility)')
    parser.add_argument('-t', '--time-limit', type=int, default=300,
                        help='Time limit for solver in seconds (default: 300)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    parser.add_argument('--soft-delta', type=float, default=0.0,
                        help='Soft relaxation: soft_min = 2*(1-soft_delta). (default: 0.0)')
    # Visualization and saving are always enabled (outputs go to MIS/outputs/5d).
    parser.add_argument('--validate-tol', type=float, default=1e-5,
                        help='Hard-valid if min_dist >= 2-validate_tol. (default: 1e-5)')

    parser.add_argument('--repair-steps', type=int, default=0,
                        help='Hard repair steps after soft MIS (0 disables). (default: 0)')
    parser.add_argument('--repair-step-size', type=float, default=0.03,
                        help='Step size for hard repair. (default: 0.03)')
    parser.add_argument('--repair-restarts', type=int, default=10,
                        help='Number of repair retries with tiny jitter. (default: 10)')
    parser.add_argument('--repair-stage-mins', type=str, default=None,
                        help='Optional multi-stage repair schedule, comma-separated. Example: "1.90,1.97,2.0". (default: disabled)')
    parser.add_argument('--repair-stage-fracs', type=str, default=None,
                        help='Optional per-stage step fractions, comma-separated. Example: "0.5,0.3,0.2". (default: even split)')

    parser.add_argument('--sampling', type=str, default='uniform',
                        choices=['uniform', 'fps'],
                        help='Candidate sampling method (default: uniform)')
    parser.add_argument('--oversample', type=int, default=20000,
                        help='When sampling=fps: number of uniform candidates before FPS (default: 20000)')
    parser.add_argument('--fps-start', type=str, default='random',
                        choices=['random', '0'],
                        help="When sampling=fps: FPS start point ('random' or '0') (default: random)")

    args = parser.parse_args()

    method = args.method

    repair_stage_mins = _parse_float_list(args.repair_stage_mins)
    repair_stage_fracs = _parse_float_list(args.repair_stage_fracs)

    run_experiment_5d(
        n_candidates=args.candidates,
        random_state=args.seed,
        method=method,
        time_limit=args.time_limit,
        soft_delta=args.soft_delta,
    visualize=True,
    save_output=True,
        sampling=args.sampling,
        oversample=args.oversample,
        fps_start=args.fps_start,
        validate_tol=args.validate_tol,
        repair_steps=args.repair_steps,
        repair_step_size=args.repair_step_size,
        repair_restarts=args.repair_restarts,
        repair_stage_mins=repair_stage_mins,
        repair_stage_fracs=repair_stage_fracs,
    )
