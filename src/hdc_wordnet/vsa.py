"""
Vector Symbolic Architecture (VSA) Primitives for HDC-WordNet.

This module provides the core Hyperdimensional Computing operations using 
bipolar dense vectors (+1, -1). 

It dynamically supports both `numpy` (CPU-bound) and `torch` (CUDA-accelerated) backends.
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Default vector dimension
D = 10000

# Global Context for Backend
class VSAConfig:
    def __init__(self):
        self.backend = "numpy"
        self.device = "cpu"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"

config = VSAConfig()

def set_backend(backend_name: str, device: str = None):
    """
    Switch the compute backend between 'numpy' and 'torch'.
    """
    if backend_name == "torch" and not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
    if backend_name not in ["numpy", "torch"]:
        raise ValueError("Backend must be 'numpy' or 'torch'.")
    
    config.backend = backend_name
    if device is not None:
        config.device = device
    else:
        if backend_name == "torch":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            config.device = "cpu"

def generate_vector(dim: int = D):
    if config.backend == "torch":
        # Generate uniform random 0 or 1, multiply by 2 to get 0 or 2, subtract 1 to get -1 or +1
        vec = torch.randint(0, 2, (dim,), device=config.device, dtype=torch.int8) * 2 - 1
        return vec
    else:
        return np.random.choice([-1, 1], size=dim).astype(np.int8)

def bind(a, b):
    # Element-wise multiplication works cleanly across both APIs
    return a * b

def bundle(vectors: list, threshold: bool = True):
    if not vectors:
        raise ValueError("Cannot bundle an empty list of vectors.")
        
    if config.backend == "torch":
        # Stack and sum
        stacked = torch.stack(vectors)
        superposition = torch.sum(stacked, dim=0)
        
        if threshold:
            zero_mask = (superposition == 0)
            num_zeros = zero_mask.sum().item()
            if num_zeros > 0:
                # Use same dtype as superposition
                random_ties = torch.randint(0, 2, (num_zeros,), device=config.device, dtype=superposition.dtype) * 2 - 1
                superposition[zero_mask] = random_ties
            return torch.sign(superposition).to(torch.int8)
        return superposition
    else:
        superposition = np.sum(vectors, axis=0)
        
        if threshold:
            zero_mask = (superposition == 0)
            num_zeros = np.sum(zero_mask)
            if num_zeros > 0:
                superposition[zero_mask] = np.random.choice([-1, 1], size=num_zeros)
            return np.sign(superposition).astype(np.int8)
        
        return superposition

def permute(a, shifts: int = 1):
    if config.backend == "torch":
        return torch.roll(a, shifts=shifts, dims=0)
    else:
        return np.roll(a, shifts)

def cosine_similarity(a, b) -> float:
    if config.backend == "torch":
        # Cast to int32 to prevent overflow, take dot product
        dot_prod = torch.dot(a.to(torch.int32), b.to(torch.int32))
        return float(dot_prod.item() / len(a))
    else:
        return float(np.dot(a.astype(np.int32), b.astype(np.int32))) / len(a)
