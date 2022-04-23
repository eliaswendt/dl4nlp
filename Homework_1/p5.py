import numpy as np
import os

# ----------------------------------------------------------------------------------------------------------------------
#               5.1 DATASET READER
# ----------------------------------------------------------------------------------------------------------------------
def read_dataset(filepath: str):

    datapoints = []

    with open(filepath) as f:
        for line in f:
            _, sentiment_str, embedding_str = line.split("\t")

            sentiment = True if sentiment_str == "label=POS" else False

            # transform embeddings to np.array
            embedding = np.array()

    return datapoints

# ----------------------------------------------------------------------------------------------------------------------
#               5.2 NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------------------

def sig(x):
    return 1 / (1 + np.exp(-x))

# ----------------------------------------------------------------------------------------------------------------------
#               5.3 TRAINING
# ----------------------------------------------------------------------------------------------------------------------

def forward(x, w):
    return sig(x.transpose * w)

def backward(w, lr, batch):
    w = 

datapoints = read_dataset("DATA/rt-polarity.dev.vecs")
# organize input in batches
# shuffle batches

w = np.random.normal(0, 1, (N, 1))



for epoch in (n_epochs):
    for X, Y in batches:

        # forward
        y = forward(x, w)
        
        # calculate loss
        loss = 

        # backward (update weights)


