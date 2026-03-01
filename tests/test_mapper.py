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

def test_definition_encoding():
    from hdc_wordnet.vsa import bind, permute, bundle
    memory = ItemMemory(dim=5000)
    # The definition of dog is: "a member of the genus Canis (probably descended from the common wolf) that has been domesticated by man since prehistoric times; occurs in many breeds"
    # word "canis" should be in the definition
    dog_synset = wn.synsets('dog')[0]
    dog_hdv = encode_synset(dog_synset, memory)
    
    rel_def = memory.get_or_create("rel_definition")
    word_canis = memory.get_or_create("word_canis")
    word_wolf = memory.get_or_create("word_wolf")
    word_computer = memory.get_or_create("word_computer") # Should NOT be in the definition
    
    # We query the dog_hdv for what is bound to its `rel_definition`.
    # H(dog) ~ bundle(..., bind(rel_definition, permute(bundle(word_1, word_2, ...))))
    # Since VSA binding over bundle distributes, bind(rel_definition, permute(bundle(w))) == bundle(bind(rel_definition, permute(w1)), ...)
    # To check if `word_canis` is part of the definition, we simulate the encoded component:
    target_component_canis = bind(rel_def, permute(word_canis))
    target_component_wolf = bind(rel_def, permute(word_wolf))
    target_component_computer = bind(rel_def, permute(word_computer))
    
    sim_canis = cosine_similarity(dog_hdv, target_component_canis)
    sim_wolf = cosine_similarity(dog_hdv, target_component_wolf)
    sim_computer = cosine_similarity(dog_hdv, target_component_computer)
    
    assert sim_canis > 0.04  # Significant connection
    assert sim_wolf > 0.02   # Significant connection
    assert abs(sim_computer) < 0.03 # Noise level 

