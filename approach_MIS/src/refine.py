"""MIS post-processing refinements.

This module adds *dimension-agnostic* continuous refinement steps that can be
used together with the MIS (discrete) approach.

Motivation
----------
Pure random sampling discretizes the sphere then solves MIS exactly on that
discrete set. Hitting a near-optimal kissing configuration by sampling alone is
very unlikely.

Instead of relaxing the physical hard-sphere constraint (min pairwise distance
>= 2 for unit-radius spheres), we keep the constraint as the correctness gate
and add a light continuous refinement step that pushes points apart and
projects them back to the sphere surface.

The refinement here mirrors the core idea from `approach_optimization/` but is
kept numpy-only so it can be used inside the MIS pipeline without requiring
PyTorch.
"""

from __future__ import annotations

import numpy as np


def project_to_sphere(points: np.ndarray, radius: float) -> np.ndarray:
    """Project points to a sphere of given radius (row-wise)."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero; if a point is exactly zero (extremely unlikely),
    # re-sample its direction.
    zero = norms.squeeze(-1) == 0
    if np.any(zero):
        points = points.copy()
        points[zero] = np.random.randn(np.sum(zero), points.shape[1])
        norms = np.linalg.norm(points, axis=1, keepdims=True)
    return radius * points / norms


def refine_repulsion_projected(
    points: np.ndarray,
    *,
    radius: float = 2.0,
    min_dist: float = 2.0,
    steps: int = 200,
    step_size: float = 0.05,
    warmup_steps: int = 100,
    riesz_s: float = 2.0,
    step_decay: float = 0.995,
    last_mile: bool = True,
    focus_k: int | None = 64,
    last_mile_band: float = 0.08,
    last_mile_lr_scale: float = 0.25,
    tol: float = 1e-6,
) -> np.ndarray:
    r"""Refine a set of points on a sphere by pushing close pairs apart.

    This performs a simple projected gradient-like update on the penalty
    $$ L = \sum_{i<j} \max(0, min\_dist - ||x_i-x_j||)^2 $$
    while always projecting back to the sphere of radius `radius`.

    Parameters
    ----------
    points:
        (N, d) points. They will be projected to the sphere initially.
    radius:
        Sphere radius for kissing number with unit balls => 2.0.
    min_dist:
        Hard sphere minimum distance (2.0 for unit radius spheres).
    steps:
        Number of refinement iterations.
    step_size:
        Update size. Larger converges faster but may oscillate.
    tol:
        Numerical epsilon for distance computations.

    Returns
    -------
    refined:
        Refined points with same shape (N, d) on the sphere.
    """
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"points must be 2D (N,d), got shape {x.shape}")

    x = project_to_sphere(x, radius)

    n, d = x.shape
    if n <= 1:
        return x

    # -------- Phase 0: Riesz warmup (global repulsion) --------
    # This encourages a more uniform cover of the sphere and prevents early
    # clumping / near-duplicates from dominating.
    # We use a simple force: f_ij ~ (x_i - x_j) / ||x_i-x_j||^(s+2)
    # (gradient of sum 1/||x_i-x_j||^s).
    lr = float(step_size)
    for _ in range(int(max(0, warmup_steps))):
        diff = x[:, None, :] - x[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + tol
        mask = ~np.eye(n, dtype=bool)
        w = np.zeros((n, n), dtype=np.float64)
        w[mask] = 1.0 / (dist[mask] ** (riesz_s + 2.0))
        g = np.einsum("ij,ijd->id", w, diff)  # gradient direction (repulsive)
        x = x + lr * g
        x = project_to_sphere(x, radius)
        lr *= step_decay

    # -------- Phase 1: Hard-sphere penalty (only for pairs < min_dist) --------
    lr = float(step_size)
    for _ in range(int(steps)):
        # Pairwise displacement: diff[i,j] = x[i]-x[j]
        diff = x[:, None, :] - x[None, :, :]  # (n,n,d)
        dist = np.linalg.norm(diff, axis=2) + tol  # (n,n)

        # Only consider i<j pairs
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)

        # Penalty only where dist < min_dist
        overlap = (min_dist - dist)
        active = (overlap > 0) & upper
        if not np.any(active):
            break

        # Optional: focus updates on the most "dangerous" pairs (closest ones).
        # This tends to help in the 1.9~2.0 jam region where most pairs are safe
        # and only a few constraints dominate.
        if focus_k is not None:
            k = int(focus_k)
            if k > 0:
                # Extract candidate pairs from the active set.
                ii, jj = np.where(active)
                if ii.size > k:
                    # Keep k pairs with smallest distances (equivalently largest overlap).
                    idx = np.argpartition(dist[ii, jj], k - 1)[:k]
                    keep = np.zeros(ii.shape[0], dtype=bool)
                    keep[idx] = True
                    active2 = np.zeros_like(active)
                    active2[ii[keep], jj[keep]] = True
                    active = active2

        # Gradient for pair (i,j): d/dx_i (min_dist - ||x_i-x_j||)^2
        # = -2*(min_dist - dist) * (x_i-x_j)/dist
        scale = np.zeros((n, n), dtype=np.float64)
        scale[active] = -2.0 * overlap[active] / dist[active]

        # Accumulate per-point gradient: g[i] = sum_j scale[i,j]*(x_i-x_j)
        g = np.einsum("ij,ijd->id", scale, diff)
        # Because we only used upper triangle, we’re missing the symmetric
        # contribution for j. Add it by subtracting transposed part.
        g = g - np.einsum("ji,jid->id", scale, diff)

        # Last-mile: as we approach the target, reduce effective LR to avoid
        # oscillation around the constraint surface.
        eff_lr = lr
        if last_mile:
            # When min_dist is close to 2.0 (or any target), we're typically
            # dealing with tiny overlaps. Use a smaller lr in that band.
            # Heuristic: if the maximum overlap is in (0, last_mile_band], scale down.
            max_ov = float(np.max(overlap[active])) if np.any(active) else 0.0
            if 0.0 < max_ov <= float(last_mile_band):
                eff_lr = lr * float(last_mile_lr_scale)

        # Take a step opposite to gradient (descent)
        x = x - eff_lr * g
        x = project_to_sphere(x, radius)

        lr *= step_decay

    return x


def pairwise_distance_quantiles(points: np.ndarray, qs=(0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)) -> dict:
    """Compute quantiles of pairwise distances (upper triangle), numpy-only."""
    x = np.asarray(points, dtype=np.float64)
    n = x.shape[0]
    if n <= 1:
        return {float(q): float("inf") for q in qs}
    diff = x[:, None, :] - x[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = dist[tri]
    # Drop NaNs defensively
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {float(q): float("nan") for q in qs}
    out = {}
    for q in qs:
        out[float(q)] = float(np.quantile(vals, q))
    return out


def min_pairwise_distance(points: np.ndarray) -> float:
    """Compute minimum pairwise distance among points (numpy-only)."""
    x = np.asarray(points, dtype=np.float64)
    n = x.shape[0]
    if n <= 1:
        return float("inf")
    diff = x[:, None, :] - x[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    # Mask self-distances without triggering invalid 0*inf operations.
    # (np.eye(n) * np.inf) creates NaN due to 0 * inf on off-diagonals.
    dist[np.eye(n, dtype=bool)] = np.inf

    m = float(np.min(dist))
    # Defensive: if something upstream created NaNs, surface it explicitly.
    if not np.isfinite(m):
        # If all distances are inf (should only happen when n<=1), keep inf.
        if np.all(np.isinf(dist)):
            return float("inf")
    return m
