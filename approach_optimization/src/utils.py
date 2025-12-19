import torch

def check_overlaps(points, min_dist=2.0, tolerance=1e-4):
    """
    Checks for overlaps in the final configuration.
    Returns the number of valid spheres (simplistic check).
    """
    N = points.shape[0]
    dist_matrix = torch.cdist(points, points, p=2)
    mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
    pairwise_dists = dist_matrix[mask]
    
    if pairwise_dists.numel() == 0:
        return {
            "min_distance": float('inf'),
            "num_overlaps": 0,
            "is_valid": True
        }

    min_d = torch.min(pairwise_dists).item()
    overlaps = (pairwise_dists < min_dist - tolerance).sum().item()
    
    return {
        "min_distance": min_d,
        "num_overlaps": overlaps,
        "is_valid": overlaps == 0
    }

def check_overlaps_batched(points, min_dist=2.0, tolerance=1e-4):
    """
    Checks for overlaps in a batch of configurations.
    points: (B, N, d)
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
        
    B = points.shape[0]
    dist_matrix = torch.cdist(points, points, p=2) # (B, N, N)
    
    mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
    
    # Calculate min distance per batch
    # Replace non-upper-triangular values with inf so they don't affect min
    dist_for_min = dist_matrix.clone()
    dist_for_min[~mask] = float('inf')
    min_dists = dist_for_min.amin(dim=(1, 2))
    
    # Count overlaps
    overlaps_mask = (dist_matrix < min_dist - tolerance) & mask
    num_overlaps = overlaps_mask.sum(dim=(1, 2))
    
    # Check for NaNs in points or distances
    # If points contain NaN, cdist returns NaN.
    # We mark any batch with NaN as invalid.
    has_nan = torch.isnan(min_dists)
    
    is_valid = (num_overlaps == 0) & (~has_nan)
    
    return {
        "min_distance": min_dists, # (B,)
        "num_overlaps": num_overlaps, # (B,)
        "is_valid": is_valid # (B,)
    }
