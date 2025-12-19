import torch

def kissing_number_loss(points, min_dist=2.0):
    """
    Calculates the loss for the kissing number problem.
    
    Args:
        points: Tensor of shape (N, d) representing the centers of the spheres.
        min_dist: The minimum required distance between sphere centers (default 2.0).
        
    Returns:
        loss: Scalar tensor representing the total overlap penalty.
    """
    # Calculate pairwise distances
    # dist_matrix[i, j] = ||points[i] - points[j]||
    dist_matrix = torch.cdist(points, points, p=2)
    
    # We only care about the upper triangle (i < j) to avoid double counting and self-distance
    # Create a mask for the upper triangle
    mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
    
    pairwise_dists = dist_matrix[mask]
    
    # Calculate overlap: max(0, min_dist - dist)
    # We want dist >= min_dist, so if dist < min_dist, penalty is (min_dist - dist)^2
    overlaps = torch.relu(min_dist - pairwise_dists)
    
    loss = torch.sum(overlaps ** 2)
    
    return loss

def centered_lennard_jones_loss(points, min_dist=2.0, p=100):
    """
    Calculates a Lennard-Jones like potential with minimum EXACTLY at min_dist.
    Form: V(r) = (min_dist/r)^(2p) - 2*(min_dist/r)^p
    
    Properties:
    1. Minimum is at r = min_dist.
    2. V(min_dist) = -1.
    3. Strong repulsion for r < min_dist.
    4. Attraction for r > min_dist (pulling towards min_dist).
    """
    dist_matrix = torch.cdist(points, points, p=2)
    mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
    r = dist_matrix[mask]
    
    ratio = min_dist / r
    term_repulsive = ratio ** (2 * p)
    term_attractive = 2 * (ratio ** p)
    
    # loss = torch.sum(term_repulsive - term_attractive + 1)  # Shift so minimum is at 0
    loss = torch.sum(term_repulsive - 1)  # Shift so minimum is at 0
    return loss
