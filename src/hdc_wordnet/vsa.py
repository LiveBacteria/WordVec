"""
Vector Symbolic Architecture (VSA) Primitives for HDC-WordNet.

This module provides the core Hyperdimensional Computing operations using 
bipolar dense vectors (+1, -1). 

Rationale & Reasoning:
----------------------
HDC relies on high-dimensional vectors (e.g., D=10,000) that are nearly orthogonal 
to one another when randomly generated. 
- `generate_vector`: Creates these foundational item vectors.
- `bind` (Multiplication): Encodes association or variable-binding (e.g., Key * Value). In bipolar vectors, this is element-wise multiplication.
- `bundle` (Addition): Represents sets or superpositions. We use majority-rule bundling thresholded back to +1/-1 to keep vectors in the same space.
- `permute` (Rotation): Encodes sequence or structure (e.g., position in a hierarchy), generally via a 1-step circular shift.
- `cosine_similarity`: The standard metric for finding the similarity between two HDVs. In bipolar spaces, normalized dot product is equivalent to bitwise Hamming distance mapping.
"""

import numpy as np

# Default vector dimension
D = 10000

def generate_vector(dim: int = D) -> np.ndarray:
    """
    Generates a random bipolar hyperdimensional vector (+1, -1).
    
    Args:
        dim: The dimensionality of the vector. defaults to `D` (10,000).
        
    Returns:
        np.ndarray: A 1D array of shape (dim,) with values in {-1, 1}.
    """
    # np.random.choice is fast enough for initialization
    # or we can use signs of normal distribution
    return np.random.choice([-1, 1], size=dim).astype(np.int8)

def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Binds two vectors together using element-wise multiplication.
    
    Binding represents the association between two concepts 
    (equivalent to XOR in binary Spatter Codes). The result is dissimilar 
    to both `a` and `b`.
    
    Args:
        a: A bipolar vector.
        b: A bipolar vector.
        
    Returns:
        np.ndarray: The bound vector (a * b).
    """
    return a * b

def bundle(vectors: list[np.ndarray], threshold: bool = True) -> np.ndarray:
    """
    Bundles a list of vectors into a single superposition vector.
    
    Bundling represents a set of concepts. Element-wise addition is performed.
    If `threshold` is True, a majority rule is applied to return the vector 
    back to the bipolar {-1, 1} space. Ties are broken randomly.
    
    Args:
        vectors: A list of bipolar vectors to bundle.
        threshold: If True, applies sign thresholding.
        
    Returns:
        np.ndarray: The bundled vector.
    """
    if not vectors:
        raise ValueError("Cannot bundle an empty list of vectors.")
        
    superposition = np.sum(vectors, axis=0)
    
    if threshold:
        # Resolve 0s (ties) randomly to +1 or -1
        zero_mask = (superposition == 0)
        num_zeros = np.sum(zero_mask)
        if num_zeros > 0:
            superposition[zero_mask] = np.random.choice([-1, 1], size=num_zeros)
            
        # Apply sign to map back to -1, +1
        return np.sign(superposition).astype(np.int8)
    
    return superposition

def permute(a: np.ndarray, shifts: int = 1) -> np.ndarray:
    """
    Permutes a vector by circularly shifting its elements.
    
    Permutation is used to encode sequence, position, or asymmetry 
    (e.g., distinguishing A->B from B->A).
    
    Args:
        a: The input vector.
        shifts: Number of positions to shift.
        
    Returns:
        np.ndarray: The permuted vector.
    """
    return np.roll(a, shifts)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the cosine similarity between two bipolar vectors.
    
    Since the vectors are bipolar (+1, -1), the magnitude of a vector of 
    dimension D is always sqrt(D).
    Thus, Cosine Sim = (a dot b) / D.
    
    Values are between -1.0 (opposite) and 1.0 (identical).
    Expected similarity between two random vectors is ~0.0.
    
    Args:
        a: A bipolar vector.
        b: A bipolar vector.
        
    Returns:
        float: Similarity score in [-1.0, 1.0].
    """
    return float(np.dot(a.astype(np.int32), b.astype(np.int32))) / len(a)
