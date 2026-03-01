"""
Tests for Vector Symbolic Architecture (VSA) Primitive Operations.
"""

import numpy as np
import pytest
from hdc_wordnet.vsa import generate_vector, bind, bundle, permute, cosine_similarity, D

def test_generate_vector():
    vec = generate_vector(100)
    assert len(vec) == 100
    # Values should only be -1 or 1
    assert np.all(np.isin(vec, [-1, 1]))

def test_orthogonality():
    # Two random vectors should be nearly orthogonal (cosine sim ~ 0)
    v1 = generate_vector(D)
    v2 = generate_vector(D)
    sim = cosine_similarity(v1, v2)
    assert abs(sim) < 0.05  # Highly likely for D=10,000

def test_bind_properties():
    # v1 * v2
    v1 = generate_vector(D)
    v2 = generate_vector(D)
    bound = bind(v1, v2)
    
    # Binding v1 with bound should recover v2
    # v1 * (v1 * v2) = (v1 * v1) * v2 = 1 * v2 = v2
    recovered_v2 = bind(v1, bound)
    assert cosine_similarity(v2, recovered_v2) == 1.0

def test_bundle_properties():
    # Bundle v1, v2, v3
    v1 = generate_vector(D)
    v2 = generate_vector(D)
    v3 = generate_vector(D)
    
    b = bundle([v1, v2, v3])
    
    # The bundled vector should be similar to all its constituents
    assert cosine_similarity(b, v1) > 0.3
    assert cosine_similarity(b, v2) > 0.3
    assert cosine_similarity(b, v3) > 0.3

def test_permute_properties():
    v1 = generate_vector(D)
    p1 = permute(v1)
    
    # A vector and its permutation should be orthogonal
    assert abs(cosine_similarity(v1, p1)) < 0.05
    
    # Permuting backwards should recover the original vector
    p1_rev = permute(p1, -1)
    assert cosine_similarity(v1, p1_rev) == 1.0
