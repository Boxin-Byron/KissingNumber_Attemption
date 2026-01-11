import numpy as np
import time
import argparse

from geometry import *
from potential import *
from dynamics import *


def run_simulation(dim, N, save_path=None, seed=43, num_step=80000):
    # ======================
    # Config
    # ======================
    DIM = dim
    SEED = seed
    num_step = num_step

    np.random.seed(SEED)

    # 初始化：随机分布
    X = np.random.randn(N, DIM)
    X = normalize(X)

    deg, num = get_min_angle_deg(X)
    print(f"Initial Min Angle: {deg:.4f}° | <60° Pairs: {num // 2}")

    # ==========================================
    # 策略调度 (Curriculum)
    # ==========================================
    schedule = [
        {'f': RieszPotential(6),   'dt': 1e-2, 'damping': 0.8, 'steps': 10000},
        {'f': RieszPotential(24),  'dt': 1e-3, 'damping': 0.8, 'steps': 10000},
        {'f': RieszPotential(64),  'dt': 1e-3, 'damping': 0.8, 'steps': 20000},
        {'f': RieszPotential(64, mode='max'), 'dt': 1e-4, 'damping': 0.8, 'steps': 20000},
        {'f': RieszPotential(100, mode='max'), 'dt': 1e-3, 'damping': 0.8, 'steps': 200000},
        {'f': RieszPotential(128, mode='max'), 'dt': 1e-4, 'damping': 0.8, 'steps': 200000},
        {'f': RieszPotential(100, mode='max'), 'dt': 1e-4, 'damping': 0.9, 'steps': num_step},
    ]

    optimizer = SphericalOptimizer(N, DIM)

    total_steps = 0
    start_time = time.time()

    for stage_idx, cfg in enumerate(schedule):
        print(f"\n--- Stage {stage_idx + 1}/{len(schedule)}: dt={cfg['dt']} ---")

        optimizer.dt = cfg['dt']
        optimizer.damping = cfg['damping']
        potential = cfg['f']

        optimizer.reset_velocity()

        for i in range(cfg['steps']):
            X = optimizer.step(X, potential)

            if i % 1000 == 0:
                deg, num = get_min_angle_deg(X)
                print(
                    f"  Step {i:6d} | Min Angle: {deg:.6f}° | <60° Pairs: {num // 2}"
                )

        total_steps += cfg['steps']

    end_time = time.time()

    # ======================
    # 结果分析
    # ======================
    final_deg, num = get_min_angle_deg(X)
    print("\n===== Simulation Finished =====")
    print(f"Total Time: {end_time - start_time:.2f}s")
    print(f"Total Steps: {total_steps}")
    print(f"Final Min Angle: {final_deg:.6f}° | <60° Pairs: {num // 2}")

    if save_path is not None:
        np.save(save_path, X)
        print(f"Configuration saved to: {save_path}")

    return X


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spherical point optimization using Riesz potentials"
    )

    parser.add_argument(
        "--dim", type=int, required=True,
        help="Dimension of the sphere (e.g. 5 for S^4)"
    )
    parser.add_argument(
        "--N", type=int, required=True,
        help="Number of points on the sphere"
    )
    parser.add_argument(
        "--save_path", type=str, default=None,
        help="Path to save final configuration (.npy). If not set, do not save."
    )
    parser.add_argument(
        "--seed", type=int, default=43,
        help="Random seed (default: 43)"
    )
    parser.add_argument(
        "--steps", type=int, default=100000,
        help="Number of steps to run (default: 1000000)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_simulation(
        dim=args.dim,
        N=args.N,
        save_path=args.save_path,
        seed=args.seed,
        num_step=args.steps
    )
