import gensim
import numpy as np
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from scipy.stats import pearsonr

# paths
simlex = "SimLex-999/SimLex-999.txt"
word2vec = "GoogleNews-vectors-negative300.bin"


def read_simlex(filepath):

    simlex = dict()

    with open(filepath) as f:
        next(f)  # skip the first line in file (csv header)
        for line in f:
            columns = line.split("\t")
            columns[-1] = columns[-1][:-1]  # cut off newline char in last column

            simlex[(columns[0], columns[1])] = float(columns[3])

    return simlex

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


def task_1_2(simlex, word_pairs):
    for word1, word2 in word_pairs:
        print(f"SimLex-999 {word1} - {word2}: {simlex[(word1, word2)]}")


def task_1_3(wv, word_pairs):
    for word1, word2 in word_pairs:
        print(f"word2vec euclidean distance {word1} - {word2}: {word2vec_distance(wv, word1, word2)}")

def task_1_4(simlex, wv, word_pairs):
    simlex_similarities = []
    euclidean_distances = []

    for word1, word2 in word_pairs:
        simlex_similarities.append(simlex[(word1, word2)])
        euclidean_distances.append(word2vec_distance(wv, word1, word2))

    print(f"pearson coefficient {word1} - {word2}: {pearsonr(simlex_similarities, euclidean_distances)[0]}") 
        

if __name__ == "__main__":
    simlex = read_simlex(simlex)
    wv = KeyedVectors.load_word2vec_format(word2vec, binary=True, limit=50000)
    word_pairs = [('happy', 'cheerful'), ('happy', 'young'),('happy', 'angry')]
  
    task_1_2(simlex, word_pairs)
    task_1_3(wv, word_pairs)
    task_1_4(simlex, wv, word_pairs)
