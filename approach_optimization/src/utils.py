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
