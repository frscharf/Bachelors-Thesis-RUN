import pickle
import numpy
import torch
import torch.nn as nn
import gensim.downloader as api
from gensim.models import KeyedVectors
import modules.data_processors
import os

embeddings = KeyedVectors.load_word2vec_format('C:/Users/fanny/Documents/Bachelor\'s Thesis/RUN/RUN-master/data/embeddings/GoogleNews-vectors-negative300.bin', binary=True)
#embeddings = api.load("word2vec-google-news-300")
#embeddings = embeddings.key_to_index
#embeddings = torch.FloatTensor(embeddings.vectors)

embeddings.vectors = embeddings.vectors[:,0:100]  # keeps just 1st 100 dims
embeddings.vector_size = 100


def embeddings_vec():
    vec = []
    #d = modules.data_processors.DataProcess(os.path.dirname(os.getcwd()) + "\data\\")
    d = modules.data_processors.DataProcess()
    for i in d.ind2word:
        try:
            vec.append(embeddings[d.ind2word[i]])
        except KeyError:
            vec.append(torch.rand(100))
    vec = torch.FloatTensor(vec)
    return vec




def sample_weights(nrow, ncol):
    """
    This is form Bengio's 2010 paper
    """

    bound = (numpy.sqrt(6.0) / numpy.sqrt(nrow+ncol) ) * 1.0
    return nn.Parameter(torch.DoubleTensor(nrow, ncol).uniform_(-bound, bound))




