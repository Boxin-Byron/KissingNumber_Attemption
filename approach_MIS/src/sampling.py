"""Sampling utilities for MIS-based kissing number search.

We only keep *dimension-agnostic* sampling primitives:

- Uniform sampling on the radius-2 sphere (centers of unit spheres tangent to
    the central unit sphere).
- FPS (Farthest Point Sampling) to downselect a more uniform subset from an
    oversampled candidate set.

We intentionally avoid dimension-specific, structure-injecting samplers (e.g.
2D evenly spaced angles, 3D Fibonacci lattice) to reduce prior bias.
"""

import numpy as np


def farthest_point_sampling(points, k, start_index=None, random_state=None, chunk_size=4096):
    """Select k points by Farthest Point Sampling (FPS).

    This is a greedy max-min diversity selection:
    pick a start point, then iteratively add the point with the largest
    distance to the current selected set.

    Notes:
    - Works in any dimension.
    - Uses squared Euclidean distances (monotonic with Euclidean).
    - Chunked distance computation keeps peak memory reasonable.

    Parameters
    ----------
    points : np.ndarray, shape (n, dim)
        Candidate points.
    k : int
        Number of points to keep.
    start_index : int or None
        If provided, use this as the first selected point.
    random_state : int or None
        Seed used only when start_index is None.
    chunk_size : int
        How many points to process per chunk when updating distances.

    Returns
    -------
    selected : np.ndarray, shape (k, dim)
        The selected points.
    indices : np.ndarray, shape (k,)
        Indices of selected points in the original array.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if k > n:
        raise ValueError(f"k cannot exceed number of points (k={k}, n={n})")

    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    if start_index is None:
        start_index = int(rng.randint(0, n))
    if not (0 <= start_index < n):
        raise ValueError("start_index out of range")

    selected_idx = np.empty(k, dtype=int)
    selected_idx[0] = start_index

    # min squared distance to the selected set for each point
    min_d2 = np.full(n, np.inf, dtype=float)

    def update_min_d2(center):
        # Update min_d2 with distances to 'center'
        # Chunk to limit memory: (chunk, dim)
        for s in range(0, n, int(chunk_size)):
            e = min(n, s + int(chunk_size))
            diff = points[s:e] - center
            d2 = np.einsum('ij,ij->i', diff, diff)
            min_d2[s:e] = np.minimum(min_d2[s:e], d2)

    update_min_d2(points[start_index])
    min_d2[start_index] = -np.inf  # never pick again

    for i in range(1, k):
        next_idx = int(np.argmax(min_d2))
        selected_idx[i] = next_idx
        update_min_d2(points[next_idx])
        min_d2[next_idx] = -np.inf

    return points[selected_idx], selected_idx


def sample_sphere_fps(n_keep, dim, radius=2.0, oversample=10000, start='random', random_state=None, chunk_size=4096):
    """Uniform oversample on sphere then keep n_keep via FPS.

    Parameters
    ----------
    n_keep : int
        How many points to return after FPS.
    dim : int
        Ambient dimension.
    radius : float
        Sphere radius.
    oversample : int
        How many uniform candidates to generate before FPS.
    start : {'random', '0'}
        FPS start point strategy.
        - 'random': random start (seeded by random_state)
        - '0': start at index 0
    random_state : int or None
        Random seed for oversampling and FPS start.
    chunk_size : int
        Chunk size for FPS distance updates.

    Returns
    -------
    points_keep : np.ndarray, shape (n_keep, dim)
    """
    oversample = int(oversample)
    n_keep = int(n_keep)
    if oversample < n_keep:
        raise ValueError(f"oversample must be >= n_keep (oversample={oversample}, n_keep={n_keep})")

    candidates = sample_sphere_uniform(n=oversample, dim=dim, radius=radius, random_state=random_state)
    start_index = 0 if str(start) == '0' else None
    keep, _ = farthest_point_sampling(
        candidates,
        k=n_keep,
        start_index=start_index,
        random_state=random_state,
        chunk_size=chunk_size,
    )
    return keep


def sample_sphere_uniform(n, dim, radius=2.0, random_state=None):
    """
    Generate n points uniformly distributed on the surface of a d-dimensional sphere.
    
    Method: Generate from standard normal distribution and normalize.
    This ensures isotropy (uniform distribution on the sphere).
    
    For Kissing Number problem with unit-radius spheres:
    - Central sphere: radius 1, at origin
    - Surrounding spheres: radius 1, centers at distance 2 from origin
    - Default radius=2.0 is correct for unit-radius spheres
    
    Parameters:
    -----------
    n : int
        Number of points to generate
    dim : int
        Dimension of the space (points will be on S^(dim-1))
    radius : float, optional
        Radius of the sphere (default: 2.0 for kissing number with r=1)
    random_state : int or None, optional
        Random seed for reproducibility
        
    Returns:
    --------
    points : np.ndarray
        Array of shape (n, dim) containing points on the sphere
        
    References:
    -----------
    Marsaglia, G. (1972). "Choosing a Point from the Surface of a Sphere"
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Sample from standard normal distribution
    points = np.random.randn(n, dim)
    
    # Normalize to unit sphere, then scale by radius
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = radius * points / norms
    
    return points

