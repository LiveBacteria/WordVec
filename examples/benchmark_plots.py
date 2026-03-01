"""
HDC-WordNet vs NLTK WordNet
Analogy Benchmarks, Accuracy, and Latency Plots (Amortized vs Non-Amortized)
"""

import time
import os
import sys
import nltk
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
import numpy as np
import pickle

from hdc_wordnet.api import build_wordvec_space, fast_semantic_search
from hdc_wordnet.vsa import bind

try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')

def run_benchmarks():
    print("--- HDC-WordNet vs NLTK WordNet: Analogy, Accuracy & Latency Benchmarks ---\n")
    
    os.makedirs("assets", exist_ok=True)
    
    # 1. Total Size Profiling
    all_synsets = list(wn.all_synsets())
    total_wn_size = len(all_synsets)
    print(f"[1] WordNet Size Profiling")
    print(f"Total Synsets in NLTK WordNet: {total_wn_size}")
    
    # Estimate HDC size: each synset is a length D (e.g., 10000) int8 array (1 byte per dim, 10KB per synset)
    # Plus ItemMemory overhead (maybe ~20,000 unique relational tokens)
    estimated_synset_mb = (total_wn_size * 10000) / (1024 * 1024)
    estimated_memory_mb = (20000 * 10000) / (1024 * 1024)
    total_estimated_mb = estimated_synset_mb + estimated_memory_mb
    
    print(f"Estimated HDC-WordNet uncompressed memory size (D=10,000, int8): ~{total_estimated_mb:.2f} MB")
    print(f"This is easily small enough to be cached entirely in RAM on a modern machine, or pickled to disk.")
    print("-" * 50)
    
    
    # 2. Benchmarking Analogical Reasoning & Scaling Output
    print("\n[2] Analogical Reasoning (Relations via Binding)")
    
    sizes = [1000, 5000, 10000, 20000]
    all_nouns = list(wn.all_synsets('n'))
    
    target_hypernym = wn.synset('domestic_animal.n.01')
    target_hypernym_name = target_hypernym.name()
    
    generation_times = []
    
    nltk_query_times = []
    
    hdc_query_times_raw_np = []
    hdc_query_times_amortized_np = []
    
    hdc_query_times_raw_pt = []
    hdc_query_times_amortized_pt = []

    accuracies = []

    # Pre-build spaces for raw latency and accuracy testing, but explicitly record amortized time
    for s in sizes:
        sub = all_nouns[:s]
        
        # Ground Truth for this subset
        nltk_match_names = set([syn.name() for syn in sub if target_hypernym in syn.hypernyms()])
        max_possible = len(nltk_match_names)
        
        # --- Numpy Run ---
        from hdc_wordnet.vsa import set_backend, permute
        set_backend("numpy")
        
        t0 = time.perf_counter()
        mem_np, sp_np = build_wordvec_space(sub, memory=None, dim=10000)
        t1 = time.perf_counter()
        amortized_cost_np = t1 - t0
        generation_times.append(amortized_cost_np)
        
        rel_hypernym_np = mem_np.get_or_create("rel_hypernym")
        target_hdv_np = mem_np.get_or_create(f"synset_{target_hypernym_name}")
        # MUST permute the target to form the correct query, matching mapper.py!
        query_np = bind(rel_hypernym_np, permute(target_hdv_np))
        
        t0_raw = time.perf_counter()
        hdc_results_np = fast_semantic_search(query_np, sp_np, top_n=max(10, max_possible))
        t1_raw = time.perf_counter()
        raw_latency_np = (t1_raw - t0_raw) * 1000 # ms
        
        hdc_query_times_raw_np.append(raw_latency_np)
        hdc_query_times_amortized_np.append(raw_latency_np + (amortized_cost_np * 1000))
        
        # --- Torch Run ---
        try:
            set_backend("torch")
            t0 = time.perf_counter()
            mem_pt, sp_pt = build_wordvec_space(sub, memory=None, dim=10000)
            
            # Flush CUDA to ensure generation is done
            if hasattr(mem_pt.memory.get(list(mem_pt.memory.keys())[0], None), 'is_cuda'):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            t1 = time.perf_counter()
            amortized_cost_pt = t1 - t0
            
            rel_hypernym_pt = mem_pt.get_or_create("rel_hypernym")
            target_hdv_pt = mem_pt.get_or_create(f"synset_{target_hypernym_name}")
            query_pt = bind(rel_hypernym_pt, permute(target_hdv_pt))
            
            # Warmup
            _ = fast_semantic_search(query_pt, sp_pt, top_n=max(10, max_possible))
            if torch.cuda.is_available(): torch.cuda.synchronize()
            
            t0_raw = time.perf_counter()
            hdc_results_pt = fast_semantic_search(query_pt, sp_pt, top_n=max(10, max_possible))
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1_raw = time.perf_counter()
            raw_latency_pt = (t1_raw - t0_raw) * 1000 # ms
            
            hdc_query_times_raw_pt.append(raw_latency_pt)
            hdc_query_times_amortized_pt.append(raw_latency_pt + (amortized_cost_pt * 1000))
            
            # Clean up VRAM
            del mem_pt, sp_pt, query_pt, hdc_results_pt
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"CUDA Not Available or Failed: {e}")
            hdc_query_times_raw_pt.append(0)
            hdc_query_times_amortized_pt.append(0)
        
        # --- NLTK Run ---
        t0_nltk = time.perf_counter()
        _ = [syn.name() for syn in sub if target_hypernym in syn.hypernyms()]
        t1_nltk = time.perf_counter()
        nltk_query_times.append((t1_nltk - t0_nltk) * 1000) # ms
        
        # --- Accuracy Calculation (from Numpy run) ---
        if max_possible > 0:
            top_hdc_names = [res[0] for res in hdc_results_np[:max_possible*2]]
            found = sum(1 for name in nltk_match_names if name in top_hdc_names)
            acc = (found / max_possible) * 100
        else:
            acc = 100.0
        accuracies.append(acc)


    # 3. Generating Plots

    # Plot A: Accuracy Progression
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, accuracies, marker='s', linestyle='-', color='green')
    plt.title('HDC Analogical Query Accuracy vs Scale')
    plt.xlabel('Search Domain Size (Number of Synsets)')
    plt.ylabel('Recall Accuracy (%) (within 2x window)')
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('assets/accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved 'assets/accuracy.png'")

    # Plot B: Split Latency Bar Chart (Amortized vs Non-Amortized vs Baseline)
    x = np.arange(len(sizes))
    width = 0.15

    plt.figure(figsize=(12, 7))
    plt.bar(x - width*2, nltk_query_times, width, label='NLTK Baseline (Graph)', color='gray')
    
    plt.bar(x - width, hdc_query_times_raw_np, width, label='HDC Raw Latency (CPU Numpy)', color='teal')
    plt.bar(x, hdc_query_times_amortized_np, width, label='HDC Amortized (CPU Numpy)', color='indigo', hatch='//')
    
    plt.bar(x + width, hdc_query_times_raw_pt, width, label='HDC Raw Latency (CUDA GPU)', color='darkorange')
    plt.bar(x + width*2, hdc_query_times_amortized_pt, width, label='HDC Amortized (CUDA GPU)', color='darkred', hatch='//')

    plt.yscale('log')
    plt.title('Search Latency Comparison (Log Scale)')
    plt.xlabel('Search Domain Size (Number of Synsets)', fontsize=12)
    plt.ylabel('Latency (ms) [Log Scale]', fontsize=12)
    plt.xticks(x, [str(s) for s in sizes])
    
    # Place legend outside to avoid obscuring bars
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('assets/query_latency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved 'assets/query_latency.png'")
    
    print("\nDone! Benchmarks and plots are complete.")

if __name__ == "__main__":
    run_benchmarks()
