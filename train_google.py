import math
import gensim
world2vector = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

print(world2vector['madrid'])
