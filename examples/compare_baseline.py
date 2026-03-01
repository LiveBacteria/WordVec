"""
Comparison Benchmark: HDC-WordNet vs NLTK WordNet
"""

import time
import nltk
from nltk.corpus import wordnet as wn
from hdc_wordnet.api import build_wordvec_space, fast_semantic_search, QueryBuilder

try:
    wn.synsets('animal')
except LookupError:
    nltk.download('wordnet')

def compare():
    print("--- HDC-WordNet vs NLTK WordNet Comparison ---\n")
    
    # Setup
    animal_synset = wn.synset('animal.n.01')
    target_hypernym_name = 'domestic_animal.n.01'
    target_hypernym = wn.synset(target_hypernym_name)
    
    # 1. Gather subset (animals)
    synsets = list(animal_synset.closure(lambda s: s.hyponyms()))
    synsets.append(animal_synset)
    print(f"Testing Domain Size: {len(synsets)} Synsets (Hyponyms of 'animal.n.01')")
    
    # -- NLTK WordNet Baseline --
    print("\n[1] NLTK WordNet: Graph Traversal Search")
    start_nltk = time.time()
    # Task: Find all synsets in our domain whose direct hypernym is 'domestic_animal.n.01'
    nltk_results = []
    for s in synsets:
        if target_hypernym in s.hypernyms():
            nltk_results.append(s.name())
    end_nltk = time.time()
    nltk_time = end_nltk - start_nltk
    print(f"Time: {nltk_time:.6f} seconds")
    print(f"Results ({len(nltk_results)} found): {nltk_results[:5]}...")

    # -- HDC-WordNet --
    print("\n[2] HDC-WordNet: Vector Search")
    # First, we need the space built (amortized cost)
    print("  -> Building HDC Space...")
    space_start = time.time()
    memory, space = build_wordvec_space(synsets, dim=10000)
    space_end = time.time()
    print(f"  -> Space built in {space_end - space_start:.4f} seconds.")
    
    qb = QueryBuilder(memory)
    # Query: Something whose hypernym is "domestic_animal.n.01"
    query = qb.relate('rel_hypernym', f'synset_{target_hypernym_name}')
    
    start_hdc = time.time()
    hdc_raw_results = fast_semantic_search(query, space, top_n=10) # Ask for top 10
    end_hdc = time.time()
    hdc_time = end_hdc - start_hdc
    print(f"Time (Search Only): {hdc_time:.6f} seconds")
    
    # Let's see how accurate the top results are compared to the ground truth
    hdc_results_names = [r[0] for r in hdc_raw_results]
    correct = sum(1 for r in hdc_results_names if r in nltk_results)
    print(f"Results (Top 10 retrieved):")
    for res, sim in hdc_raw_results:
        is_match = " (MATCH)" if res in nltk_results else ""
        print(f"  {res}: {sim:.4f}{is_match}")
        
    print(f"\nAccuracy (Top 10): {correct} out of {min(len(nltk_results), 10)} actual ground truths were in the top 10.")

if __name__ == "__main__":
    compare()
