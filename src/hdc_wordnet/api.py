"""
HDC-WordNet API Module

Provides functions for batch-converting Synsets into a searchable "WordVec" space 
and querying that space using similarity metrics. Also includes a `QueryBuilder`
to construct semantic inquiries (e.g., analogical reasoning shapes).

Rationale & Reasoning:
----------------------
The core of Vector Symbolic Architectures is comparing composed vectors (queries) 
against stored superpositions (Item Memory or Clean-up memory). 
`build_wordvec_space` acts as the ingestion pipeline.
`semantic_search` provides O(N) linear scan similarity search.
For larger N, one could port `semantic_search` to Faiss or Annoy, but since vector
operations via dot-product on small N (WordNet synsets ~117k) are relatively fast in NumPy,
we stick to standard numpy matrix operations.
"""

import numpy as np
from typing import Dict, List, Tuple
from hdc_wordnet.mapper import ItemMemory, encode_synset
from hdc_wordnet.vsa import cosine_similarity, bind, bundle

def build_wordvec_space(synsets: list, memory: ItemMemory = None, dim: int = 10000) -> Tuple[ItemMemory, Dict[str, np.ndarray]]:
    """
    Ingests a list of WordNet Synsets and generates their composite HDVs.
    
    Args:
        synsets: Iterable of NLTK Synsets.
        memory: Pre-existing ItemMemory. If None, created fresh.
        dim: Dimensionality of vectors if creating a new memory.
        
    Returns:
        tuple(ItemMemory, Dictionary mapping synset names to HDVs)
    """
    if memory is None:
        memory = ItemMemory(dim=dim)
        
    space = {}
    for syn in synsets:
        hdv = encode_synset(syn, memory)
        space[syn.name()] = hdv
        
    return memory, space


def semantic_search(query: np.ndarray, space: Dict[str, np.ndarray], top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Finds the most similar entries in the WordVec space to the provided query HDV.
    
    Args:
        query: Composed HDV representing the query.
        space: The encoded space built by `build_wordvec_space`.
        top_n: Number of nearest neighbors to return.
        
    Returns:
        A list of tuples: (synset_name, similarity_score)
    """
    results = []
    # Using simple loop. Can be vectorized easily:
    # keys = list(space.keys())
    # matrix = np.array(list(space.values()), dtype=np.int32)
    # q_int32 = query.astype(np.int32)
    # scores = np.dot(matrix, q_int32) / len(query)
    # ... but loop logic handles smaller subsets just fine and clearly.
    
    for name, hdv in space.items():
        sim = cosine_similarity(query, hdv)
        results.append((name, sim))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def fast_semantic_search(query, space: Dict, top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Vectorized O(N) search across the entire dictionary. Supports both Numpy and PyTorch.
    """
    if not space:
        return []
    
    from hdc_wordnet.vsa import config
    names = list(space.keys())
    
    if config.backend == "torch":
        import torch
        # Shape: (N, D)
        matrix = torch.stack(list(space.values())).to(torch.float32)
        q_vec = query.to(torch.float32)
        
        # Matrix multiply (CUDA requires floats for mv/mm)
        sim_scores = torch.mv(matrix, q_vec) / float(len(query))
        
        # Find top N indices (descending)
        top_indices = torch.argsort(sim_scores, descending=True)[:top_n]
        return [(names[idx.item()], float(sim_scores[idx].item())) for idx in top_indices]
    else:
        # Shape: (N, D)
        matrix = np.array(list(space.values()), dtype=np.int32)
        q_vec = query.astype(np.int32)
        
        # Dot product broadcasts: (N, D) dot (D,) -> (N,)
        sim_scores = np.dot(matrix, q_vec) / float(len(query))
        
        # Find top N indices
        # argsort is ascending, so take the last top_n and reverse
        top_indices = np.argsort(sim_scores)[-top_n:][::-1]
        
        return [(names[idx], float(sim_scores[idx])) for idx in top_indices]



class QueryBuilder:
    """
    A helper class for ergonomically constructing complex queries.
    """
    def __init__(self, memory: ItemMemory):
        self.memory = memory
        
    def get_base(self, token: str) -> np.ndarray:
        """Retrieves raw HDV for a token."""
        return self.memory.get_or_create(token)
        
    def relate(self, relation: str, target_token: str) -> np.ndarray:
        """
        Creates a structured relation bound to a target, ensuring directed edge
        encoding via permutation (rho).
        E.g., QueryBuilder.relate('rel_hypernym', 'synset_animal.n.01')
        """
        from hdc_wordnet.vsa import permute
        r_hdv = self.get_base(relation)
        t_hdv = self.get_base(target_token)
        return bind(r_hdv, permute(t_hdv))
        
    def compose(self, components: List[np.ndarray], threshold: bool = True) -> np.ndarray:
        """
        Bundles components into a single query vector.
        """
        return bundle(components, threshold=threshold)
