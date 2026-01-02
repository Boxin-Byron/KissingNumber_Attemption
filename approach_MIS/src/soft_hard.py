"""Soft→Hard pipeline utilities for MIS-based kissing number search.

The MIS approach discretizes the sphere into a candidate set, builds a conflict
graph, then solves a Maximum Independent Set (MIS).

In low dimensions (and in particular for 2D), strictly sampling candidates and
requiring the strict hard constraint (min pairwise distance >= 2) can make the
optimal size hard to hit.

This module implements a *two-stage* strategy inspired by continuous
optimization workflows:

1) Soft stage: relax the geometric constraint by a small delta and solve MIS.
2) Hard stage (repair): starting from the selected soft solution, run a small
   continuous refinement to push points apart back to the strict constraint.

Important:
  - Only the *final* repaired configuration that passes strict validation should
    be interpreted as a kissing-number *lower bound*.
  - The soft-stage MIS size is an auxiliary signal (seed quality), not a bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.spatial.distance import pdist

from .graph import build_conflict_graph
from .mis_solver import solve_mis
from .refine import refine_repulsion_projected


def min_pairwise_distance(points: np.ndarray) -> float:
    """Compute min pairwise distance using pdist (fast for small N)."""
    x = np.asarray(points, dtype=np.float64)
    if x.shape[0] <= 1:
        return float("inf")
    return float(np.min(pdist(x)))


@dataclass
class SoftHardResult:
    soft_min_dist: float
    hard_min_dist: float
    soft_mis_size: int
    soft_mis_nodes: list[int]
    repaired_points: np.ndarray | None
    repaired_min_dist: float | None
    is_hard_valid: bool
    repair_attempts: int


def solve_soft_then_repair(
    *,
    points: np.ndarray,
    hard_min_dist: float = 2.0,
    soft_delta: float = 0.0,
    graph_epsilon: float = 1e-6,
    mis_method: str = "auto",
    time_limit: float = 300,
    repair_steps: int = 0,
    repair_step_size: float = 0.02,
    repair_warmup_steps: int | None = None,
    repair_stage_mins: list[float] | None = None,
    repair_stage_fracs: list[float] | None = None,
    repair_restarts: int = 1,
    validate_tol: float = 1e-5,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> SoftHardResult:
    """Solve MIS under a relaxed constraint then try to repair to strict hard constraint.

    Parameters
    ----------
    points:
        Candidate points (N,d) on sphere (radius should already be 2 for r=1).
    hard_min_dist:
        Strict target min distance (2.0 for unit spheres).
    soft_delta:
        Relative relaxation amount. soft_min = hard_min * (1-soft_delta).
        Set to 0 to disable soft stage (soft==hard).
    repair_steps:
        If >0, run continuous refinement on the selected set.
    repair_restarts:
        Number of retry attempts for repair. Each retry adds a tiny random jitter
        before refinement to escape local jams.
    repair_stage_mins:
        Optional multi-stage schedule for the repair min_dist.
        Example: [1.90, 1.97, 2.00]. If provided, the repair will run stages in
        order and only the final stage corresponds to the strict hard constraint.
        This often helps when soft_delta is large.
    repair_stage_fracs:
        Fractions for splitting repair_steps across stages. Must sum to 1.
        If omitted, steps are split evenly across stages.
    validate_tol:
        Final acceptance is min_dist >= hard_min_dist - validate_tol.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if not (0.0 <= soft_delta < 1.0):
        raise ValueError(f"soft_delta must be in [0,1), got {soft_delta}")

    soft_min_dist = float(hard_min_dist * (1.0 - soft_delta))

    # --- soft stage (graph + MIS) ---
    G_soft = build_conflict_graph(points, min_dist=soft_min_dist, epsilon=graph_epsilon)
    soft_nodes, soft_size, _, _ = solve_mis(
        G_soft, method=mis_method, time_limit=time_limit, verbose=verbose
    )

    if verbose:
        print(
            f"[soft→hard] soft_min_dist={soft_min_dist:.6f}, "
            f"soft MIS size={soft_size}"
        )

    hard_target = float(hard_min_dist)
    attempts = 0
    repaired = None
    repaired_min = None
    hard_ok = False

    if repair_steps <= 0:
        return SoftHardResult(
            soft_min_dist=soft_min_dist,
            hard_min_dist=hard_target,
            soft_mis_size=int(soft_size),
            soft_mis_nodes=list(map(int, soft_nodes)),
            repaired_points=None,
            repaired_min_dist=None,
            is_hard_valid=False,
            repair_attempts=0,
        )

    base_sel = np.asarray(points)[np.asarray(soft_nodes, dtype=int)]
    if base_sel.shape[0] <= 1:
        return SoftHardResult(
            soft_min_dist=soft_min_dist,
            hard_min_dist=hard_target,
            soft_mis_size=int(soft_size),
            soft_mis_nodes=list(map(int, soft_nodes)),
            repaired_points=base_sel,
            repaired_min_dist=min_pairwise_distance(base_sel),
            is_hard_valid=True,
            repair_attempts=0,
        )

    if repair_warmup_steps is None:
        repair_warmup_steps = max(50, repair_steps // 2)

    # Build stage schedule (default: single stage at hard_target)
    if repair_stage_mins is None:
        stage_mins = [hard_target]
    else:
        stage_mins = [float(v) for v in repair_stage_mins]
        if len(stage_mins) == 0:
            stage_mins = [hard_target]
        # Ensure a sane final target.
        if abs(stage_mins[-1] - hard_target) > 1e-9:
            stage_mins = list(stage_mins) + [hard_target]
        # Basic monotonicity check (non-decreasing)
        for a, b in zip(stage_mins, stage_mins[1:]):
            if b + 1e-12 < a:
                raise ValueError(f"repair_stage_mins must be non-decreasing, got {stage_mins}")

    n_stages = len(stage_mins)
    if repair_stage_fracs is None:
        stage_fracs = [1.0 / n_stages] * n_stages
    else:
        stage_fracs = [float(v) for v in repair_stage_fracs]
        if len(stage_fracs) != n_stages:
            raise ValueError(
                f"repair_stage_fracs must have same length as stage_mins ({n_stages}), got {len(stage_fracs)}"
            )
        s = float(sum(stage_fracs))
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"repair_stage_fracs must sum to positive value, got {stage_fracs}")
        stage_fracs = [v / s for v in stage_fracs]

    # Allocate steps per stage (ensure sum==repair_steps and each stage >=1 when repair_steps>0)
    if repair_steps > 0:
        raw = [max(0, int(round(repair_steps * f))) for f in stage_fracs]
        # Fix rounding drift
        drift = int(repair_steps - sum(raw))
        if drift != 0:
            raw[-1] += drift
        # Ensure at least 1 step per stage when possible
        # (helps avoid degenerate stages)
        for i in range(n_stages):
            if raw[i] == 0 and repair_steps >= n_stages:
                raw[i] = 1
        # Re-adjust to total
        drift2 = int(repair_steps - sum(raw))
        raw[-1] += drift2
        stage_steps = [int(max(1, v)) for v in raw]
        # Final fix again
        drift3 = int(repair_steps - sum(stage_steps))
        stage_steps[-1] += drift3
    else:
        stage_steps = [0] * n_stages

    # --- hard stage (repair) ---
    for k in range(int(max(1, repair_restarts))):
        attempts += 1

        # Tiny jitter to help escape symmetric jams; keep it very small relative
        # to radius=2.
        jitter = 1e-3 * rng.standard_normal(size=base_sel.shape)
        sel0 = base_sel + jitter

        sel = sel0
        for stage_i, (stage_min, stage_nsteps) in enumerate(zip(stage_mins, stage_steps), start=1):
            if verbose and n_stages > 1:
                print(
                    f"[soft→hard]  stage {stage_i}/{n_stages}: "
                    f"target_min={stage_min:.6f}, steps={stage_nsteps}"
                )
            sel = refine_repulsion_projected(
                sel,
                radius=2.0,
                min_dist=float(stage_min),
                steps=int(stage_nsteps),
                step_size=float(repair_step_size),
                warmup_steps=int(repair_warmup_steps) if stage_i == 1 else 0,
                riesz_s=2.0,
                step_decay=0.995,
            )

        md = min_pairwise_distance(sel)
        if verbose:
            print(f"[soft→hard] repair attempt {attempts}: min_dist={md:.6f}")

        if md >= hard_target - float(validate_tol):
            repaired = sel
            repaired_min = md
            hard_ok = True
            break

        # Track best attempt even if it fails
        if repaired_min is None or md > repaired_min:
            repaired = sel
            repaired_min = md

    return SoftHardResult(
        soft_min_dist=soft_min_dist,
        hard_min_dist=hard_target,
        soft_mis_size=int(soft_size),
        soft_mis_nodes=list(map(int, soft_nodes)),
        repaired_points=repaired,
        repaired_min_dist=float(repaired_min) if repaired_min is not None else None,
        is_hard_valid=bool(hard_ok),
        repair_attempts=int(attempts),
    )
