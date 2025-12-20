import torch
import itertools

def solve_candidates_batched(basis_points, device='cuda'):
    """
    Given a batch of basis sets, find the candidate points that touch all basis points and the origin.
    
    Args:
        basis_points: Tensor of shape (B, D-1, D) where B is batch size (number of combinations),
                      D is dimension. Each row is a point p_i.
                      
    Returns:
        candidates: Tensor of shape (B, 2, D). There are usually 2 solutions per basis set.
                    Contains NaNs if no solution exists.
    """
    B, K, D = basis_points.shape
    assert K == D - 1, f"Need D-1 points to fix a candidate in D dimensions (plus origin). Got {K} for D={D}."
    
    # We need to solve:
    # 1. x . p_i = 2  for all i (Linear constraints)
    # 2. ||x||^2 = 4  (Spherical constraint)
    
    # Linear system: P x = b
    # P is (B, D-1, D), x is (B, D, 1), b is (B, D-1, 1) full of 2s.
    P = basis_points
    b = torch.full((B, K, 1), 2.0, device=device, dtype=basis_points.dtype)
    
    # We want to find x = x_p + alpha * v
    # x_p is a particular solution to P x = b
    # v is a vector in the null space of P (direction orthogonal to all p_i)
    
    # 1. Find Null Space Vector v
    # Since P has rank D-1 (usually), the null space is 1D.
    # v is the cross product of the rows of P (generalized to D dimensions).
    # Or simpler: v is the last row of V^T from SVD, or we can use QR.
    # For efficiency in PyTorch, we can use torch.linalg.cross for D=3.
    # For general D, we can use QR decomposition. P^T = Q R. 
    # The last column of Q is orthogonal to columns of P^T (rows of P).
    
    # P_trans: (B, D, D-1)
    P_trans = P.transpose(1, 2)
    
    # Full QR decomposition: P^T = Q R
    # Q: (B, D, D), R: (B, D, D-1)
    # The last column of Q corresponds to the null space of P.
    try:
        Q, R = torch.linalg.qr(P_trans, mode='complete')
        v = Q[:, :, -1:] # (B, D, 1)
    except RuntimeError:
        # Handle singular matrices or other errors
        return torch.full((B, 2, D), float('nan'), device=device)

    # 2. Find Particular Solution x_p
    # We can use the Moore-Penrose pseudoinverse: x_p = P^+ b
    # Or since we have Q, R from P^T = Q R => P = R^T Q^T
    # P x = R^T Q^T x = b
    # This might be complicated to reconstruct. Let's just use lstsq or pinv.
    # pinv is robust.
    P_pinv = torch.linalg.pinv(P) # (B, D, D-1)
    x_p = torch.bmm(P_pinv, b)    # (B, D, 1)
    
    # 3. Solve for alpha
    # ||x_p + alpha * v||^2 = 4
    # ||x_p||^2 + 2 alpha (x_p . v) + alpha^2 ||v||^2 = 4
    # Note: x_p is in row space, v is in null space => x_p . v should be 0 theoretically.
    # Let's verify or just solve the full quadratic equation A alpha^2 + B alpha + C = 0
    
    v_sq = torch.sum(v ** 2, dim=1)       # (B, 1)
    xp_sq = torch.sum(x_p ** 2, dim=1)    # (B, 1)
    xp_dot_v = torch.sum(x_p * v, dim=1)  # (B, 1)
    
    A = v_sq
    B_coef = 2 * xp_dot_v
    C = xp_sq - 4.0
    
    # Delta = B^2 - 4AC
    delta = B_coef**2 - 4 * A * C
    
    # Filter negative delta (no intersection)
    # delta is (B, 1). We want a 1D boolean mask of size (B,)
    mask_valid = (delta >= 0).view(-1)
    
    # Prepare output
    candidates = torch.full((B, 2, D), float('nan'), device=device, dtype=basis_points.dtype)
    
    if mask_valid.any():
        sqrt_delta = torch.sqrt(delta[mask_valid])
        A_valid = A[mask_valid]
        B_valid = B_coef[mask_valid]
        
        alpha1 = (-B_valid + sqrt_delta) / (2 * A_valid)
        alpha2 = (-B_valid - sqrt_delta) / (2 * A_valid)
        
        # Reconstruct x
        # x = x_p + alpha * v
        # Need to index x_p and v with mask
        x_p_valid = x_p[mask_valid] # (N_valid, D, 1)
        v_valid = v[mask_valid]     # (N_valid, D, 1)
        
        sol1 = x_p_valid + alpha1.unsqueeze(1) * v_valid
        sol2 = x_p_valid + alpha2.unsqueeze(1) * v_valid
        
        # Fill into candidates
        # sol1 shape: (N_valid, D, 1) -> (N_valid, D)
        candidates[mask_valid, 0, :] = sol1.squeeze(2)
        candidates[mask_valid, 1, :] = sol2.squeeze(2)
        
    return candidates

def check_validity_batched(candidates, existing_points, min_dist=1.999, epsilon=1e-5):
    """
    Check if candidates overlap with any existing points.
    
    Args:
        candidates: (B, 2, D)
        existing_points: (N, D) - The current configuration
        
    Returns:
        valid_mask: (B, 2) boolean tensor. True if candidate is valid.
    """
    # candidates: (B, 2, D) -> (B*2, D)
    B, _, D = candidates.shape
    flat_cands = candidates.view(-1, D)
    
    # Compute distances to all existing points
    # dists: (B*2, N)
    dists = torch.cdist(flat_cands, existing_points)
    
    # Check if any distance is too small
    # We allow distance to be exactly 2.0 (touching), so we check < 2.0 - epsilon
    # But wait, the candidate is generated to touch D-1 points. 
    # Those distances will be exactly 2.0 (within float error).
    # We need to ensure it doesn't touch OTHERS with dist < 2.0.
    
    # Using a slightly smaller threshold to be safe against float errors
    # If dist is 1.999999, it's fine. If 1.9, it's bad.
    # So threshold should be e.g. 1.99
    threshold = min_dist - epsilon
    
    # min_dists: (B*2,)
    min_dists_to_existing, _ = dists.min(dim=1)
    
    is_valid = min_dists_to_existing > threshold
    
    # Also check for NaNs (failed geometry solve)
    is_not_nan = ~torch.isnan(flat_cands).any(dim=1)
    
    final_valid = is_valid & is_not_nan
    
    return final_valid.view(B, 2)
