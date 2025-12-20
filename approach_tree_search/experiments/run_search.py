import sys
import os
import torch
import argparse
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search import BeamSearch

def run_search(dim, beam_width=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    searcher = BeamSearch(dim=dim, beam_width=beam_width, device=device)
    
    start_time = time.time()
    best_config = searcher.run(max_steps=100)
    duration = time.time() - start_time
    
    final_n = best_config.shape[0]
    print(f"Finished D={dim}. Found N={final_n} in {duration:.2f}s")
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", f"{dim}d")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"tree_search_d{dim}_n{final_n}.pt")
    torch.save(best_config, save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--beam', type=int, default=50)
    args = parser.parse_args()
    
    run_search(args.dim, args.beam)
