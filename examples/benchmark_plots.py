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
    
    # Target definition for tests
    # Query: Find things whose hypernym is "domestic_animal.n.01"
    target_hypernym = wn.synset('domestic_animal.n.01')
    target_hypernym_name = target_hypernym.name()
    
    generation_times = []
    
    nltk_query_times = []
    hdc_query_times_raw = []        # Just matrix search
    hdc_query_times_amortized = []  # Space build + matrix search

    accuracies = [] # % of true matches found in HDC's top N

    mem = None
    spaces = []
    
    # Pre-build spaces for raw latency and accuracy testing, but explicitly record amortized time
    for s in sizes:
        sub = all_nouns[:s]
        
        # Ground Truth for this subset
        nltk_match_names = set([syn.name() for syn in sub if target_hypernym in syn.hypernyms()])
        max_possible = len(nltk_match_names)
        
        # Amortized Generation Time
        t0 = time.perf_counter()
        mem, sp = build_wordvec_space(sub, memory=mem, dim=10000)
        t1 = time.perf_counter()
        amortized_cost = t1 - t0
        generation_times.append(amortized_cost)
        spaces.append(sp)
        
        # We simulate amortized latency for *one* query as: BuildSpace + Search 
        # (Though practically, we build once and search M times)
        
        # HDC Setup: Create the relational query
        rel_hypernym = mem.get_or_create("rel_hypernym")
        target_hdv = mem.get_or_create(f"synset_{target_hypernym_name}")
        query = bind(rel_hypernym, target_hdv)
        
        # A) HDC Raw Query Latency (Just the search matrix op)
        t0_raw = time.perf_counter()
        hdc_results = fast_semantic_search(query, sp, top_n=max(10, max_possible))
        t1_raw = time.perf_counter()
        raw_latency = (t1_raw - t0_raw) * 1000 # ms
        
        hdc_query_times_raw.append(raw_latency)
        hdc_query_times_amortized.append(raw_latency + (amortized_cost * 1000)) # ms + ms
        
        # B) NLTK Query latency
        t0_nltk = time.perf_counter()
        _ = [syn.name() for syn in sub if target_hypernym in syn.hypernyms()]
        t1_nltk = time.perf_counter()
        nltk_query_times.append((t1_nltk - t0_nltk) * 1000) # ms
        
        # C) Accuracy Calculation
        # How many of the ground truth NLTK names were in the top HDC results?
        if max_possible > 0:
            top_hdc_names = [res[0] for res in hdc_results[:max_possible*2]] # Give it a 2x leeway window for fuzzy similarity
            found = sum(1 for name in nltk_match_names if name in top_hdc_names)
            acc = (found / max_possible) * 100
        else:
            acc = 100.0 # Trivial if no true answers exist in subset
        
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
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, nltk_query_times, width, label='NLTK Baseline (Graph Traversal)', color='gray')
    plt.bar(x, hdc_query_times_raw, width, label='HDC Raw Latency (Matrix Search Only)', color='teal')
    
    # We plot amortized on a logarithmic scale because it drastically dwarfs the raw latency
    plt.bar(x + width, hdc_query_times_amortized, width, label='HDC Amortized (Generation + Search)', color='indigo', hatch='//')

    plt.yscale('log') # Log scale is critical to see raw latency alongside amortized!
    plt.title('Search Latency Comparison (Log Scale)')
    plt.xlabel('Search Domain Size (Number of Synsets)')
    plt.ylabel('Latency (ms) [Log Scale]')
    plt.xticks(x, [str(s) for s in sizes])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('assets/query_latency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved 'assets/query_latency.png'")
    
    print("\nDone! Benchmarks and plots are complete.")

if __name__ == "__main__":
    run_benchmarks()
