import torch
import time
from .geometry import solve_candidates_batched, check_validity_batched

class BeamSearch:
    def __init__(self, dim, beam_width=100, device='cuda'):
        self.dim = dim
        self.beam_width = beam_width
        self.device = device
        
        # Initialize with a simplex (D+1 points mutually touching)
        # This is a safe starting configuration for any D.
        # Actually, for Kissing Number, we start with D points touching the central sphere?
        # No, we start with just 1 point, or D points.
        # Let's start with a simple seed: 1 point at (2, 0, ...).
        # Or better: A max-clique we know exists.
        # For D=2, start with 2 points.
        # For general D, we can start with 1 point.
        pass

    def get_initial_state(self):
        # Construct a seed of D points using Cholesky decomposition
        # We want K points v_1...v_K such that ||v_i||=2 and ||v_i - v_j||=2
        # This implies <v_i, v_j> = 2 for i != j, and 4 for i == j.
        
        K = self.dim  # Start with D points. This is always possible for D >= 2.
        
        # Gram matrix
        G = torch.full((K, K), 2.0, device=self.device)
        G.fill_diagonal_(4.0)
        
        # Cholesky: G = L L^T
        # L will be (K, K). The rows of L are our points.
        # Since we are in D dimensions and K=D, this works perfectly.
        # If K < D, we pad with zeros.
        try:
            L = torch.linalg.cholesky(G)
        except RuntimeError:
            # Fallback if Cholesky fails (shouldn't for this matrix)
            print("Cholesky failed for initialization. Fallback to single point.")
            p0 = torch.zeros(self.dim, device=self.device)
            p0[0] = 2.0
            return p0.unsqueeze(0)
            
        # L is lower triangular.
        # Shape (K, K).
        # If K < D, we need to pad columns.
        # If K == D, it's fine.
        
        if K < self.dim:
            padding = torch.zeros((K, self.dim - K), device=self.device)
            points = torch.cat([L, padding], dim=1)
        else:
            points = L
            
        return points

    def generate_candidates_for_state(self, state_points):
        """
        Generate all valid candidates for a single state.
        state_points: (N, D)
        """
        N, D = state_points.shape
        if N < D - 1:
            # Not enough points to form a basis
            return torch.empty((0, D), device=self.device)
            
        # 1. Generate all combinations of D-1 points
        # We use torch.combinations (available in recent pytorch) or manual indexing
        indices = torch.combinations(torch.arange(N, device=self.device), r=D-1)
        # indices: (K, D-1) where K = N choose D-1
        
        # Limit K if it's too large?
        # For N=40, D=5, K ~ 90,000. It fits in memory.
        
        # Gather basis points
        # basis_sets: (K, D-1, D)
        basis_sets = state_points[indices]
        
        # 2. Solve geometry
        # candidates: (K, 2, D)
        candidates = solve_candidates_batched(basis_sets, device=self.device)
        
        # 3. Check validity
        # valid_mask: (K, 2)
        valid_mask = check_validity_batched(candidates, state_points)
        
        # 4. Filter
        valid_cands = candidates[valid_mask] # (M, D)
        
        return valid_cands

    def run(self, max_steps=100):
        # Optimized Beam Search with full batching
        
        # Initial state: (1, N_init, D)
        initial_state = self.get_initial_state().unsqueeze(0)
        current_beam = initial_state # Tensor (Beam, N, D)
        
        print(f"Starting Batched Beam Search (D={self.dim}, Width={self.beam_width})...")
        
        for step in range(max_steps):
            B, N, D = current_beam.shape
            print(f"Step {step}: Beam size {B}, N = {N}")
            
            if N < D - 1:
                print("Error: N < D-1, cannot generate candidates.")
                break
                
            # 1. Generate Combinations Indices
            # We want all combinations of D-1 points from N points.
            # indices: (K, D-1)
            indices = torch.combinations(torch.arange(N, device=self.device), r=D-1)
            K = indices.shape[0]
            
            # 2. Gather Basis Sets for ALL beams
            # current_beam: (B, N, D)
            # We want to select K subsets for each of the B beams.
            # basis_sets: (B, K, D-1, D)
            
            # Expand beam to (B, K, N, D) - too big? No.
            # Use fancy indexing.
            # current_beam[:, indices] -> (B, K, D-1, D)
            basis_sets = current_beam[:, indices, :]
            
            # Flatten for solver: (B*K, D-1, D)
            basis_sets_flat = basis_sets.view(B * K, D - 1, D)
            
            # 3. Solve Geometry
            # candidates_flat: (B*K, 2, D)
            candidates_flat = solve_candidates_batched(basis_sets_flat, device=self.device)
            
            # Reshape: (B, K, 2, D)
            candidates = candidates_flat.view(B, K, 2, D)
            
            # 4. Check Validity
            # We need to check each candidate against its OWN beam's points.
            # candidates: (B, K, 2, D) -> flatten to (B, K*2, D)
            cands_per_beam = candidates.view(B, K * 2, D)
            
            # existing_points: (B, N, D)
            # We want dists between (B, K*2, D) and (B, N, D)
            # torch.cdist supports batching!
            # dists: (B, K*2, N)
            dists = torch.cdist(cands_per_beam, current_beam)
            
            # Check min dist > 1.99
            # min_dists: (B, K*2)
            min_dists, _ = dists.min(dim=2)
            
            # Check NaNs
            # is_nan: (B, K*2)
            is_nan = torch.isnan(cands_per_beam).any(dim=2)
            
            # valid_mask: (B, K*2)
            valid_mask = (min_dists > 1.999) & (~is_nan)
            
            # 5. Gather Valid Candidates and Score Them
            # We want to pick the best candidates across the entire beam (or per beam?)
            # Standard Beam Search: Pool all valid next states from all parents, pick top W.
            
            # Construct new states?
            # A new state is (N+1, D).
            # We have potentially B * K * 2 candidates.
            # We can't construct all of them if B*K*2 is huge.
            # We should score them first.
            
            # Heuristic Score: Number of contacts (dist < 2.01)
            # We already computed dists: (B, K*2, N)
            # contacts: (B, K*2)
            contacts = (dists < 2.01).sum(dim=2)
            
            # We also want to prioritize candidates that are valid.
            # Set score of invalid candidates to -1
            scores = contacts.float()
            scores[~valid_mask] = -1.0
            
            # Flatten scores to (B * K * 2)
            flat_scores = scores.view(-1)
            
            # Find top k indices
            # If we have no valid candidates, stop.
            if scores.max() < 0:
                print("Search ended: No more valid moves.")
                break
                
            # Number of candidates to keep
            num_keep = min(self.beam_width, (flat_scores >= 0).sum().item())
            if num_keep == 0:
                break
                
            top_scores, top_indices = torch.topk(flat_scores, k=num_keep)
            
            # Reconstruct states from indices
            # index = b * (K*2) + k2
            # b = index // (K*2)
            # k2 = index % (K*2)
            
            K2 = K * 2
            beam_indices = top_indices.div(K2, rounding_mode='floor') # (num_keep,)
            cand_indices = top_indices % K2                           # (num_keep,)
            
            # Gather parent states
            # parents: (num_keep, N, D)
            parents = current_beam[beam_indices]
            
            # Gather new points
            # cands_per_beam: (B, K*2, D)
            # We need to select [beam_indices, cand_indices]
            new_points = cands_per_beam[beam_indices, cand_indices] # (num_keep, D)
            
            # Concatenate
            # next_beam: (num_keep, N+1, D)
            next_beam = torch.cat([parents, new_points.unsqueeze(1)], dim=1)
            
            current_beam = next_beam
            
        return current_beam[0]

