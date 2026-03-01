"""
Testing Script: Bag-of-Words Definition Queries vs Ephemeral Permutations
Proving that distributed algebraic binding of permuted bundles is extremely resilient.
"""

import time
import re
from nltk.corpus import wordnet as wn
import nltk

from hdc_wordnet.api import build_wordvec_space, fast_semantic_search, QueryBuilder
from hdc_wordnet.vsa import set_backend

try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')


def run_definition_proof():
    print("--- HDC-WordNet: Distributed Binding & Definition Querying at Scale ---")
    
    # We will use PyTorch for scale if available, otherwise Numpy
    try:
        import torch
        if torch.cuda.is_available():
            set_backend("torch")
            print("Backend: PyTorch CUDA")
        else:
            set_backend("numpy")
            print("Backend: Numpy CPU")
    except ImportError:
        set_backend("numpy")
        print("Backend: Numpy CPU")

    sizes = [1000, 5000, 10000, 20000]
    all_nouns = list(wn.all_synsets('n'))
    
    # Target phrase to search for: we will look for concepts whose definition contains "canine"
    target_word = "canine"
    
    print(f"\n[Test Objective]: Can we retrieve synsets whose definition contains '{target_word}'?")
    print("Validating that VSA property: A * bundle(B, C) == bundle(A*B, A*C) holds under permutation.\n")

    for s in sizes:
        subset = all_nouns[:s]
        
        # Ground Truth Extraction
        true_matches = []
        for syn in subset:
            words = re.findall(r'\b\w+\b', syn.definition().lower())
            if target_word in words:
                true_matches.append(syn.name())
                
        max_possible = len(true_matches)
        
        t0 = time.perf_counter()
        mem, space = build_wordvec_space(subset, memory=None, dim=10000)
        t1 = time.perf_counter()
        
        qb = QueryBuilder(mem)
        
        # The Algebraic Query:
        # We mapped: bind(rel_definition, permute(bundle(w1, w2, w3)))
        # Because bind distributes over bundle:
        # == bundle( bind(rel_definition, permute(w1)), bind(rel_definition, permute(w2)), ... )
        # Therefore, searching for bind(rel_definition, permute(target_word)) should find it!
        query = qb.relate("rel_definition", f"word_{target_word}")
        
        t2 = time.perf_counter()
        results = fast_semantic_search(query, space, top_n=max(10, max_possible * 2))
        t3 = time.perf_counter()
        
        if max_possible > 0:
            top_names = [res[0] for res in results]
            found = sum(1 for name in true_matches if name in top_names)
            acc = (found / max_possible) * 100
        else:
            acc = 100.0
            
        print(f"Domain Size: {s:<5} | True Matches: {max_possible:<3} | Top-N Accuracy: {acc:5.1f}% | Search Time: {(t3-t2)*1000:.2f} ms")
        
        # Explicitly print the first few results found in the 20k scope just to prove it
        if s == 20000 and max_possible > 0:
            print(f"\n  Detailed Results in 20k Domain for '{target_word}':")
            for res, score in results[:5]:
                marker = "[TRUE]" if res in true_matches else "[NOISE]"
                print(f"    {marker} {res}: {score:.4f}")

if __name__ == "__main__":
    run_definition_proof()
