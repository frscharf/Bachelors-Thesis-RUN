import pickle
import numpy
import torch
import torch.nn as nn
import gensim.downloader as api
from gensim.models import KeyedVectors
import modules.data_processors
from sklearn.decomposition import PCA
import os

embeddings = KeyedVectors.load_word2vec_format('C:/Users/fanny/Documents/Bachelor\'s Thesis/RUN/RUN-master/data/embeddings/GoogleNews-vectors-negative300.bin', binary=True)
#embeddings = api.load("word2vec-google-news-300")
#embeddings = embeddings.key_to_index
#embeddings = torch.FloatTensor(embeddings.vectors)

#embeddings.vectors = embeddings.vectors[:,0:100]  # keeps just 1st 100 dims
#embeddings.vectors = embeddings.vectors
#embeddings.vector_size = 100


def embeddings_vec():
    vec = []
    #d = modules.data_processors.DataProcess(os.path.dirname(os.getcwd()) + "\data\\")
    d = modules.data_processors.DataProcess()
    for i in d.ind2word:
        try:
            vec.append(embeddings[d.ind2word[i]])
        except KeyError:
            #vec.append(torch.rand(100))
            vec.append(torch.normal(0, 1, size=(300,1)))
    vec = torch.FloatTensor(vec)
    # Start of PPA new code
    #PCA to get Top Components
    pca = PCA(n_components = 300)
    vec = vec - vec.mean()
    vec = pca.fit_transform(vec)
    U1 = pca.components_

    z = []

    # Removing Projections on Top Components
    for i, x in enumerate(vec):
        for u in U1[0:3]:
            x = x - numpy.dot(u.transpose(),x) * u
        z.append(x)

    z = numpy.asarray(z)
    #End of ppa
    #Start of PCA
    s = PCA(n_components=100)
    new_vec = z - z.mean()
    new_vec = s.fit_transform(new_vec)

    # PCA to get Top Components
    pca = PCA(n_components=100)
    vec = new_vec - new_vec.mean()
    vec = pca.fit_transform(vec)
    U1 = pca.components_

    z = []

    # Removing Projections on Top Components
    for i, x in enumerate(vec):
        for u in U1[0:3]:
            x = x - numpy.dot(u.transpose(), x) * u
        z.append(x)

    z = numpy.asarray(z)
    new_vec = torch.FloatTensor(z)
    return new_vec




def sample_weights(nrow, ncol):
    """
    This is from Bengio's 2010 paper
    """

    bound = (numpy.sqrt(6.0) / numpy.sqrt(nrow+ncol) ) * 1.0
    return nn.Parameter(torch.DoubleTensor(nrow, ncol).uniform_(-bound, bound))




