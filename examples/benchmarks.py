"""
Benchmarking script for HDC-WordNet

Measures the generation time, memory foot-print, and querying speed of the 
WordVec space at scale.
"""

import time
import nltk
from nltk.corpus import wordnet as wn
from hdc_wordnet.api import build_wordvec_space, fast_semantic_search, QueryBuilder

try:
    wn.synsets('entity')
except LookupError:
    nltk.download('wordnet')

def run_benchmarks():
    print("--- HDC-WordNet Benchmarks ---")
    
    # 1. Scaling Ingestion
    print("\n1. Preprocessing and Generation Scales")
    # Get all nouns
    print("Gathering all noun synsets in WordNet...")
    all_nouns = list(wn.all_synsets('n'))
    total_ns = len(all_nouns)
    print(f"Total Nouns: {total_ns}")
    
    # Ingest 10k, 50k, All
    cutoffs = [1000, 10000, 40000, total_ns]
    
    memory = None
    spaces = []
    
    for c in cutoffs:
        subset = all_nouns[:c]
        start_t = time.time()
        memory, space = build_wordvec_space(subset, memory=memory, dim=10000)
        end_t = time.time()
        spaces.append(space)
        print(f"Ingested {c} synsets in {end_t - start_t:.3f} seconds. (Base Memory Size: {len(memory.memory)} HDVs)")
        
    space_full = spaces[-1]
    print(f"\nFinal Space keys: {len(space_full)}")
    
    # 2. Searching
    print("\n2. Vectorized Search Benchmark")
    qb = QueryBuilder(memory)
    # create a struct query
    query = qb.relate('rel_hypernym', 'synset_animal.n.01')
    
    for c, space in zip(cutoffs, spaces):
        start_s = time.time()
        _ = fast_semantic_search(query, space, top_n=5)
        end_s = time.time()
        
        print(f"Queried space of {c} elements in {end_s - start_s:.4f} seconds.")
        
if __name__ == "__main__":
    run_benchmarks()
