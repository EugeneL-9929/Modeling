from gensim.models import Word2Vec
# CBOW - continuous bag of words (input: )
# SG - skip gram 
import gensim.downloader as api
# loading pre-trained word2vec model
# wv = api.load('word2vec-google-news-300')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# t-distrbuted stochastic neighbour embedding
from sklearn.preprocessing import MinMaxScaler
# scale points to [-1, 1]

