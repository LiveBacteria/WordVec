"""
HDC WordNet Mapper

Transforms WordNet structures into Hyperdimensional Bipolar Vectors.
This module maintains the `ItemMemory` for base tokens and provides functions
to compose complex semantic episodes from Synsets.
"""

import numpy as np
from nltk.corpus import wordnet as wn
import nltk

from typing import Dict, List, Optional
from hdc_wordnet.vsa import generate_vector, bind, bundle, permute, cosine_similarity

# Ensure WordNet is available
try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')

class ItemMemory:
    """
    Maintains a mapping of string tokens to their foundational HDVs.
    
    Tokens include:
    - POS tags: `pos_n`, `pos_v`, `pos_a`, `pos_r`
    - Relational tags: `rel_hypernym`, `rel_hyponym`, `rel_meronym`, etc.
    - Base concepts or lemmas if required.
    """
    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.memory: Dict[str, np.ndarray] = {}
        
    def get_or_create(self, token: str) -> np.ndarray:
        """Retrieves an existing HDV for the token or creates a new one."""
        if token not in self.memory:
            self.memory[token] = generate_vector(self.dim)
        return self.memory[token]
    
    def add(self, token: str, hdv: np.ndarray):
        """Explicitly adds an HDV to the memory."""
        self.memory[token] = hdv

    def find_nearest(self, query: np.ndarray, top_n: int = 5) -> List[tuple[str, float]]:
        """
        Finds the top N tokens in memory most similar to the query vector.
        """
        results = []
        for token, hdv in self.memory.items():
            sim = cosine_similarity(query, hdv)
            results.append((token, sim))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]


def encode_synset(synset, item_memory: ItemMemory) -> np.ndarray:
    """
    Encodes a Synset into an HDV.
    
    Base encoding structure:
    `H(Synset) = bundle( POS, bind(rel_hypernym, H(Hypernym)), ... )`
    
    For simplicity in this base wrapper, we encode:
    1. The core identity of the synset name itself.
    2. Its Part Of Speech (POS).
    3. Its immediate hypernyms.
    4. Its lemmas.
    
    Args:
        synset: An nltk.corpus.reader.wordnet.Synset object.
        item_memory: Reference to the ItemMemory space.
        
    Returns:
        np.ndarray: The composed HDV for the synset.
    """
    # Base identity
    name_token = f"synset_{synset.name()}"
    base_hdv = item_memory.get_or_create(name_token)
    
    components = [base_hdv]
    
    # 1. Part of Speech
    pos_token = f"pos_{synset.pos()}"
    pos_hdv = item_memory.get_or_create(pos_token)
    components.append(pos_hdv)
    
    # 2. Hypernyms (using bind and permute for directionality)
    rel_hypernym = item_memory.get_or_create("rel_hypernym")
    for hypernym in synset.hypernyms():
        hyp_token = f"synset_{hypernym.name()}"
        hyp_hdv = item_memory.get_or_create(hyp_token)
        # We apply `permute` to the target to encode directionality (X -> Y).
        # This prevents symmetric confounding (A * B == B * A).
        components.append(bind(rel_hypernym, permute(hyp_hdv)))
        
    # 3. Lemmas (using bind and permute)
    rel_lemma = item_memory.get_or_create("rel_lemma")
    for lemma in synset.lemmas():
        lem_token = f"lemma_{lemma.name()}"
        lem_hdv = item_memory.get_or_create(lem_token)
        components.append(bind(rel_lemma, permute(lem_hdv)))
        
    return bundle(components)
