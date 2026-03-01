"""
Demo/Benchmark script showing how to build a subset of WordNet into an HDC space
and resolve queries.
"""

import time
import nltk
from nltk.corpus import wordnet as wn
from hdc_wordnet.mapper import ItemMemory
from hdc_wordnet.api import build_wordvec_space, fast_semantic_search, QueryBuilder

# Ensure WordNet
try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')

def run_demo():
    print("Gathering Synsets...")
    # Gather a subset: everything in the noun 'animal' hierarchy.
    animal = wn.synset('animal.n.01')
    synsets = list(animal.closure(lambda s: s.hyponyms()))
    synsets.append(animal)
    
    print(f"Total Synsets to encode: {len(synsets)}")
    
    # Pre-build vector space
    start_t = time.time()
    memory, space = build_wordvec_space(synsets, dim=10000)
    end_t = time.time()
    print(f"Encoding {len(synsets)} synsets took {end_t - start_t:.3f} seconds.")
    print(f"Item Memory size: {len(memory.memory)} unique base vectors created.")
    
    # Create an arbitrary query
    # E.g., we want something whose hypernym is "domestic_animal.n.01"
    qb = QueryBuilder(memory)
    # The relation for hypernym points UP in WordNet.
    # Therefore, the target's hypernym IS the given token.
    # When encoded, `encode_synset` bounds `rel_hypernym` to its actual hypernyms.
    
    query1 = qb.relate('rel_hypernym', 'synset_domestic_animal.n.01')
    
    print("\n--- Semantic Search: Animals whose Hypernym is 'domestic_animal.n.01' ---")
    start_s = time.time()
    results = fast_semantic_search(query1, space, top_n=5)
    end_s = time.time()
    print(f"Search took {end_s - start_s:.4f} seconds.")
    for res, sim in results:
        print(f"  {res}: {sim:.4f}")
    # Example 1: "What is the hypernym of 'dog'?"
    # We want to find X such that: bind(rel_hypernym, permute(X)) is in dog's vector.
    # Therefore X ≈ bundle of items whose permuted selves were bound to rel_hypernym.
    # The `QueryBuilder.relate` fundamentally handles this directed edge: it permutes the target.
    print("\n--- Example 1: Directed Semantic Search ---")
    query_hypernym = qb.relate("rel_hypernym", "synset_dog.n.01")
    
    results = fast_semantic_search(query_hypernym, space, top_n=3)
    for res, score in results:
        print(f"Match: {res} | Sim: {score:.4f}")

    # Another Query: What is closest to 'dog.n.01'?
    query_dog = space['dog.n.01']
    print("\n--- Semantic Search: Closest structured vectors to 'dog.n.01' ---")
    results_dog = fast_semantic_search(query_dog, space, top_n=5)
    for res, sim in results_dog:
        print(f"  {res}: {sim:.4f}")

    # Example 2: Querying via natural language definition semantics
    print("\n--- Example 2: Querying by Gloss/Definition Semantic Content ---")
    # Finding an animal whose definition contains the concept "canine"
    # Query: What has `word_canine` in its definition?
    query_def = qb.relate("rel_definition", "word_canine")
    results_def = fast_semantic_search(query_def, space, top_n=5)
    for res, score in results_def:
        print(f"Match: {res} | Sim: {score:.4f}")

if __name__ == "__main__":
    run_demo()
