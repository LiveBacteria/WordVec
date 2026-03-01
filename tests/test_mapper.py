"""
Tests for WordNet Mapper and HDC Encoder.
"""

import numpy as np
from hdc_wordnet.mapper import ItemMemory, encode_synset
from hdc_wordnet.vsa import cosine_similarity
from nltk.corpus import wordnet as wn
import nltk

# Ensure tests have WordNet
try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')

def test_item_memory_creation():
    memory = ItemMemory(dim=100)
    
    vec1 = memory.get_or_create("pos_n")
    vec2 = memory.get_or_create("pos_v")
    
    assert len(vec1) == 100
    assert len(vec2) == 100
    # They should be different
    assert not np.array_equal(vec1, vec2)
    # They should retrieve consistently
    vec1_again = memory.get_or_create("pos_n")
    assert np.array_equal(vec1, vec1_again)

def test_encode_synset():
    memory = ItemMemory(dim=5000)
    dog_synset = wn.synsets('dog')[0]
    cat_synset = wn.synsets('cat')[0]
    
    dog_hdv = encode_synset(dog_synset, memory)
    cat_hdv = encode_synset(cat_synset, memory)
    
    assert len(dog_hdv) == 5000
    assert len(cat_hdv) == 5000
    
    # Are they somehow similar? (they share pos_n and both have hypernym relation encoding)
    sim = cosine_similarity(dog_hdv, cat_hdv)
    
    # We expect some similarity because they share 'pos_n' and their structural encoding
    # shares the hypernym binding, but they differ in base tokens.
    assert sim > 0.05
