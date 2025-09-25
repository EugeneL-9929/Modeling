from gensim.models import Word2Vec
# CBOW - continuous bag of words (input: context output: center)
# SG - skip gram (input: center output: context) 
import gensim.downloader as api
# loading pre-trained word2vec model
# wv = api.load('word2vec-google-news-300')
from gensim.models import KeyedVectors
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# t-distrbuted stochastic neighbour embedding
from sklearn.preprocessing import MinMaxScaler
# scale points to [-1, 1]

