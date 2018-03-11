from gensim.models import KeyedVectors
from scipy import stats
import numpy as np
vecfile = 'GoogleNews-vectors-negative300.bin'
vecs = KeyedVectors.load_word2vec_format(vecfile, binary=True)

test_words = ["bad", "good", "terrible", "young", "diffident", "unqualified",
              "qualified", "happy", "unhappy", "sure", "confident", "delighted"]