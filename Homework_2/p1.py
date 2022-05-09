import gensim
import numpy as np
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from scipy.stats import pearsonr

# Data Reader

simlex = "SimLex-999/SimLex-999.txt"
word2vec = "GoogleNews-vectors-negative300.bin"


def read_simlex(filepath):

    data = dict()

    with open(filepath) as f:
        next(f)  # skip the first line in file (csv header)
        for line in f:
            columns = line.split("\t")
            columns[-1] = columns[-1][:-1]  # cut off newline char in last column

            data[(columns[0], columns[1])] = float(columns[3])

    return data

def sim_score(data, word1, word2):
    return data[(word1, word2)]


# 1.3 Ranking Based on Word2Vec

def calculate_euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def word2vec_distance(wv, word1, word2):

    try:
        word1_vec = wv.get_vector(word1, norm=True)
    except KeyError:
        #print("could not find word1")
        word1_vec = np.zeros(300)
    try:
        word2_vec = wv.get_vector(word2, norm=True)
    except KeyError:
        #print("could not find word2")
        word2_vec = np.zeros(300)

    return calculate_euclidean_distance(word1_vec, word2_vec)


# 1.4 Correlation

def compute_correlation(data, wv, word_pairs):
    simlex_similarities = []
    euclidean_distances = []

    for word1, word2 in word_pairs:
        #print(f"{word1} - {word2} -> {compute_correlation(data, wv, word1, word2)}")
        simlex_similarities.append(sim_score(data, word1, word2))
        euclidean_distances.append(word2vec_distance(wv, word1, word2))

    return pearsonr(simlex_similarities, euclidean_distances)
        




if __name__ == "__main__":
    data = read_simlex(simlex)
    wv = KeyedVectors.load_word2vec_format(word2vec, binary=True, limit=50000)
    word_pairs = [('happy', 'cheerful'), ('happy', 'young'),('happy', 'angry')]
    print(compute_correlation(data, wv, word_pairs)[0])
    print('Pair "happy, cheerful": ', sim_score(data, "happy", "cheerful"))
    print('Pair "happy, young": ', sim_score(data, "happy", "young"))
    print('Pair "happy, angry": ', sim_score(data, "happy", "angry"))
