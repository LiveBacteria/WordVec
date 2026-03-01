"""
Test script to demonstrate the issue with symmetric binding on directed graphs
and verify the permutation (rho) fix.
"""

from hdc_wordnet.api import build_wordvec_space, fast_semantic_search, QueryBuilder
from hdc_wordnet.vsa import bind, permute, cosine_similarity
from nltk.corpus import wordnet as wn
import nltk

try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')

def run_test():
    print("--- Testing Directionality in HDC-WordNet ---")
    
    # We want a very small, controlled space:
    # animal.n.01 -> domestic_animal.n.01 -> dog.n.01
    synsets = [
        wn.synset('animal.n.01'),
        wn.synset('domestic_animal.n.01'),
        wn.synset('dog.n.01')
    ]
    
    mem, space = build_wordvec_space(synsets, dim=10000)
    qb = QueryBuilder(mem)
    
    # A symmetric bind means:  relation * target = encoding
    # If dog has hypernym 'domestic_animal', its encoding contains: rel_hypernym * domestic_animal
    
    rel_hypernym = mem.get_or_create("rel_hypernym")
    domestic_animal_hdv = mem.get_or_create("synset_domestic_animal.n.01")
    
    print("\n[Query 1] What is the hypernym of 'dog'? (Symmetric Query)")
    # If we bound symmetrically: dog * rel_hypernym ≈ domestic_animal
    # But because binding is commutative (A*B = B*A), if A's hypernym is B (encoded rel*B),
    # Does B's hyponym relation easily confuse the math?
    
    # Let's test the current mapping which uses permute!
    # If `mapper.py` encodes: bind(rel_hypernym, permute(hypernym_hdv)), then it is NOT symmetric!
    
    qb = QueryBuilder(mem)
    # The new correct query for "something whose hypernym is X" should account for permutation.
    # If encoded as: bundle( ... bind(rel, permute(X)) ... )
    # Then binding by rel extracts permute(X).
    
    query = qb.relate('rel_hypernym', 'synset_domestic_animal.n.01')
    
    results = fast_semantic_search(query, space, top_n=3)
    for r, s in results:
        print(f"  {r}: {s:.4f}")

if __name__ == "__main__":
    run_test()
