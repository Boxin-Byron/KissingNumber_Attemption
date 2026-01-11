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
            
            # --- Optimization 2: Basis Filtering ---
            # If any pair in the basis has distance > 4.0, they cannot form a valid slot.
            # dist = ||a-b||. a, b in D-1.
            # We can use cdist on each (D-1, D) block.
            # basis_sets_flat: (M, D-1, D)
            # pdist: (M, D-1, D-1)
            pdist = torch.cdist(basis_sets_flat, basis_sets_flat)
            # max_dist: (M,)
            max_pdist = pdist.amax(dim=(1, 2))
            # Valid basis mask
            basis_valid_mask = max_pdist <= 4.0 + 1e-5
            
            # If no basis is valid, break early? No, maybe some are valid.
            # We only solve for valid bases to save time, or just mask the output?
            # Solving is expensive, so let's filter. But filtering changes shape.
            # To keep batching simple, we can just zero out invalid ones and let solver produce nan?
            # Or we can solve only valid ones. Let's solve all but rely on validity check.
            # Actually, let's just use the mask to invalidate candidates later.
            # This is simpler for tensor shapes.
            
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
            
            # Expand basis mask to (B, K, 2)
            # basis_valid_mask: (B*K) -> (B, K)
            basis_mask_expanded = basis_valid_mask.view(B, K).unsqueeze(2).expand(B, K, 2).reshape(B, K*2)
            
            # valid_mask: (B, K*2)
            valid_mask = (min_dists > 1.999) & (~is_nan) & basis_mask_expanded
            
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
            
            # --- Optimization 3: Heuristic Diversity & Lattice Bias ---
            # Secondary score: Sum of inverse distances (Coulomb-like) to prefer
            # points that are far from existing points (maximize future space).
            # dists: (B, K*2, N). Avoid div by zero (dists are approx >= 2).
            # potential = sum(1/d)
            potential = (1.0 / (dists + 1e-6)).sum(dim=2)
            
            # Lattice Bias (PackingStar Insight):
            # Optimal packings often have specific cosine values (e.g. 0.5 for 60 degrees).
            # cos(theta) = dot(u, v) / (|u||v|). Here |u|=|v|=2, so dot = 4 cos(theta).
            # dist^2 = |u-v|^2 = |u|^2 + |v|^2 - 2 dot = 4 + 4 - 2 dot = 8 - 8 cos(theta).
            # For 60 deg (touching), cos=0.5, dist^2 = 8 - 4 = 4 => dist = 2.
            # For 90 deg, cos=0, dist^2 = 8 => dist = 2.828.
            # For 120 deg, cos=-0.5, dist^2 = 8 + 4 = 12 => dist = 3.464.
            # For 180 deg, cos=-1, dist^2 = 16 => dist = 4.
            
            # We favor candidates that form "clean" structural angles with OTHER points.
            # We measure how close the cosines are to {0.5, 0.0, -0.5, -1.0}.
            # But wait, dists are already computed.
            # Touching neighbors (dist ~ 2.0) are already handled by 'contacts'.
            # We care about NON-touching neighbors.
            
            # dists: (B, K*2, N)
            # Convert to cosines: dot = (8 - dist^2) / 2 = 4 - 0.5 * dist^2.
            # Wait, dot product of vectors of length 2 is 4 * cos.
            # So cos = (4 - 0.5 * dist^2) / 4 = 1 - 0.125 * dist^2.
            
            cosines = 1.0 - 0.125 * (dists ** 2)
            
            # Define target "nice" cosines (Lattice-like)
            target_cosines = torch.tensor([0.5, 0.25, 0.0, -0.25, -0.5, -1.0], device=self.device)
            # 0.25 is for Leech lattice related structures (PackingStar mentions +/- 1/4).
            
            # Calculate "Lattice Alignnment Score"
            # For each neighbor, find min distance to a target cosine
            # cosines.unsqueeze(-1): (B, K*2, N, 1)
            # target_cosines: (T,)
            # diff: (B, K*2, N, T)
            cos_diff = torch.abs(cosines.unsqueeze(-1) - target_cosines)
            min_diff, _ = cos_diff.min(dim=-1) # (B, K*2, N)
            
            # We want to minimize min_diff.
            # Score bonus = sum(exp(-lambda * min_diff^2))
            lattice_score = torch.exp(-10.0 * (min_diff**2)).sum(dim=2)
            
            # Combined score: 
            # 1. Contacts (Primary): Fill holes.
            # 2. Lattice Score (Secondary): Prefer structured packing.
            # 3. Potential (Tertiary): Break ties by maximizing separation.
            
            # Weights need tuning. Contacts is roughly int (3~10). Lattice is roughly N. Potential is roughly N/2.
            # score = 1000 * contacts + 1.0 * lattice_score - 0.1 * potential
            
            scores = contacts.float() * 1000.0 + lattice_score * 2.0 - potential * 0.1
            
            scores[~valid_mask] = -1e9
            
            # --- Optimization 1: Deduplication (Greedy Selection) ---
            # Instead of topk, we select candidates that are distinct.
            
            # Flatten everything
            flat_scores = scores.view(-1)
            flat_cands = cands_per_beam.view(-1, D)
            flat_parents = current_beam.repeat_interleave(K * 2, dim=0).view(-1, N, D)
            # Note: flat_parents is huge (B*K*2, N, D). Don't materialize if possible.
            # We only need parent INDICES to reconstruct.
            parent_indices = torch.arange(B, device=self.device).repeat_interleave(K * 2)
            
            # Filter to only valid
            valid_indices_flat = torch.nonzero(flat_scores > -1e8).squeeze()
            
            if valid_indices_flat.numel() == 0:
                print("Search ended: No more valid moves.")
                break
                
            valid_scores = flat_scores[valid_indices_flat]
            
            # Sort valid candidates by score
            sorted_score_indices = torch.argsort(valid_scores, descending=True)
            sorted_global_indices = valid_indices_flat[sorted_score_indices]
            
            next_beam_list = []
            selected_next_points = []
            
            # Limit check count to speed up (e.g. check top 5*Width)
            check_limit = min(len(sorted_global_indices), self.beam_width * 10)
            
            for idx in sorted_global_indices[:check_limit]:
                if len(next_beam_list) >= self.beam_width:
                    break
                    
                cand_point = flat_cands[idx]
                parent_idx = parent_indices[idx]
                
                # Deduplication Check:
                # Is quite similar to any point already selected for THIS parent?
                # Actually, strictly speaking, we want to avoid duplicate STATES.
                # State = {Parent Points} U {Cand Point}.
                # Since parents are likely distinct, we just need to ensure we don't 
                # add the same point to the same parent multiple times.
                
                is_duplicate = False
                
                # Check against other candidates selected for the SAME parent
                # This requires tracking selected points per parent.
                # Or simpler: Check against ALL selected points? No, that prevents symmetrical branches.
                
                # Let's just create a unique key? Float keys are bad.
                # Distance check is robust.
                
                # To be efficient, we can check dist against `selected_next_points` 
                # ONLY if `parent_idx` matches.
                
                for i, (p_idx, p_pt) in enumerate(selected_next_points):
                    if p_idx == parent_idx:
                        if torch.norm(p_pt - cand_point) < 1e-4:
                            is_duplicate = True
                            break
                            
                if is_duplicate:
                    continue
                    
                # Add to new beam
                parent_state = current_beam[parent_idx]
                new_state = torch.cat([parent_state, cand_point.unsqueeze(0)], dim=0)
                next_beam_list.append(new_state)
                selected_next_points.append((parent_idx, cand_point))
            
            if len(next_beam_list) == 0:
                break
                
            current_beam = torch.stack(next_beam_list)
            
            # --- Optimization 4: Relaxation (PackingStar inspired) ---
            # "Game-theoretic" idea: We placed a point (Player 1), now we "Correct" it (Player 2).
            # We run a short burst of gradient descent on the WHOLE beam.
            # This allows points to drift away from exact tangent slots to pack tighter.
            # Only do this if N is large enough to matter? Or always?
            # Always is better for quality.
            
            # Relax the beam
            # current_beam: (B, N+1, D)
            current_beam = self.relax_beam(current_beam)
            
        return current_beam[0]

    def relax_beam(self, beam_points, steps=50, lr=0.01):
        """
        Runs batched gradient descent to relax the configurations in the beam.
        Drifts points to maximize separation.
        """
        B, N, D = beam_points.shape
        
        # Clone and enable grad
        points = beam_points.detach().clone()
        points.requires_grad_(True)
        
        optimizer = torch.optim.SGD([points], lr=lr, momentum=0.9)
        
        # Target distance
        # We want to maintain >= 2.0.
        # Loss = sum(ReLU(2.0 - dist)^2)
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            # Compute pairwise distances
            # pdist: (B, N, N)
            pdist = torch.cdist(points, points)
            
            # Mask diagonal
            mask = torch.eye(N, device=self.device).bool().unsqueeze(0).expand(B, N, N)
            pdist = pdist.masked_fill(mask, 10.0) # Ignore self
            
            # Only penalize overlaps < 2.0
            # We want slightly expansive force to fix "almost" overlaps?
            # Let's start with hard constraint.
            overlaps = torch.relu(2.0 - pdist)
            
            # Loss
            loss = (overlaps ** 2).sum()
            
            if loss < 1e-5:
                break
                
            loss.backward()
            optimizer.step()
            
            # Re-normalize to sphere surface
            with torch.no_grad():
                points.data = 2.0 * points.data / torch.norm(points.data, dim=2, keepdim=True)
                
        return points.detach()


