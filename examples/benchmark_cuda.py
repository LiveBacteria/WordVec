"""
HDC-WordNet CUDA Benchmark
Comparing `numpy` (CPU) vs `torch` (CUDA) performance on WordVec search.
"""

import time
import os
import nltk
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
import numpy as np

from hdc_wordnet.vsa import set_backend
from hdc_wordnet.api import build_wordvec_space, fast_semantic_search

try:
    wn.synsets('animal')
except LookupError:
    nltk.download('wordnet')

def run_cuda_benchmark():
    print("--- HDC-WordNet CUDA GPU Benchmark ---")
    
    os.makedirs("assets", exist_ok=True)
    
    # 1. Base Setup
    all_nouns = list(wn.all_synsets('n'))
    sizes = [1000, 10000, 20000, 40000, 60000]
    
    numpy_search_times = []
    torch_search_times = []
    
    # Prebuild Spaces for both backends (we just care about search latency here)
    print("\nCaching Item Memories...")
    # Dummy query token
    
    for size in sizes:
        subset = all_nouns[:size]
        
        # --- NUMPY TEST ---
        set_backend("numpy")
        mem_np, space_np = build_wordvec_space(subset, dim=10000)
        query_np = mem_np.get_or_create("rel_hypernym")  # Arbitrary query
        
        # Measure Numpy Search
        t0 = time.perf_counter()
        _ = fast_semantic_search(query_np, space_np, top_n=5)
        t1 = time.perf_counter()
        numpy_search_times.append((t1 - t0) * 1000) # ms
        
        # --- TORCH (CUDA) TEST ---
        try:
            set_backend("torch")
        except Exception as e:
            print(f"Failed to load Torch backend: {e}")
            return
            
        mem_pt, space_pt = build_wordvec_space(subset, dim=10000)
        query_pt = mem_pt.get_or_create("rel_hypernym")
        
        # Measure Torch Search (Run once to warmup CUDA graphs)
        _ = fast_semantic_search(query_pt, space_pt, top_n=5)
        
        # Actual measurement
        if hasattr(query_pt, 'is_cuda') and query_pt.is_cuda:
            import torch
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        _ = fast_semantic_search(query_pt, space_pt, top_n=5)
        
        if hasattr(query_pt, 'is_cuda') and query_pt.is_cuda:
            import torch
            torch.cuda.synchronize()
            
        t1 = time.perf_counter()
        torch_search_times.append((t1 - t0) * 1000) # ms

        print(f"Domain Size: {size} | Numpy: {numpy_search_times[-1]:.2f} ms | Torch: {torch_search_times[-1]:.2f} ms")


    # Plotting
    x = np.arange(len(sizes))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, numpy_search_times, width, label='NumPy Matrix Dot (CPU)', color='teal')
    plt.bar(x + width/2, torch_search_times, width, label='PyTorch Matrix Dot (GPU)', color='darkorange')
    
    plt.title('HDC-WordNet: CPU vs GPU Search Latency')
    plt.xlabel('Search Domain Size (Number of Synsets)')
    plt.ylabel('Latency (ms)')
    plt.xticks(x, [str(s) for s in sizes])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('assets/cuda_benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n  -> Saved 'assets/cuda_benchmark.png'")

if __name__ == "__main__":
    run_cuda_benchmark()
