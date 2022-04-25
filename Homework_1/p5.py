from re import U
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

            y = 1.0 if sentiment_str == "label=POS" else 0.0

            embedding = [float(x) for x in embedding_str.split(" ")]
            embedding.append(1.0)
            x = np.array(embedding)

            datapoints.append((x, y))

    return datapoints

# ----------------------------------------------------------------------------------------------------------------------
#               5.2 NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------------------

def sig(x):
    return 1 / (1 + np.exp(-x))

def sig_deriv(x):
    return sig(x)*(1-sig(x))

# ----------------------------------------------------------------------------------------------------------------------
#               5.3 TRAINING
# ----------------------------------------------------------------------------------------------------------------------

def forward(x, w):
    print(f"x.shape={x.shape}, w.shape={w.shape}")
    return sig(x.T.dot(w)) 

def backward(w, lr, batch):

    sum = np.zeros(len(batch))

    for x, y in batch:

        part1 = (sig(x.T.dot(w)) - y)
        part2 = sig_deriv(x.T.dot(w))

        print(f"part1.shape={part1.shape}, part2.shape={part2.shape}, x.shape={x.shape}")

        sum += part1 * part2 * x

    print(f"sum.shape={sum.shape}, w.shape={w.shape}")

    w -= (lr / len(batch)) * sum



def loss(y, y_hat):
    return (y_hat - y)**2 

def train():
    # initialize weights randomly (with seed)
    np.random.seed(100)


    datapoints_dev = read_dataset("DATA/rt-polarity.dev.vecs")
    datapoints_test = read_dataset("DATA/rt-polarity.test.vecs")
    datapoints_train = read_dataset("DATA/rt-polarity.train.vecs")

    datapoints = datapoints_dev

    # organize input in batches
    # shuffle batches

    N = len(datapoints[0][0]) # number of lines/datapoints
    w = np.random.normal(0, 1, (N, 1)) 

    lr = .01
    n_epochs = 50
    batch_size = 10

    n_batches = np.ceil(len(datapoints) / batch_size)
    print(f"n_batches={n_batches}")

    for epoch in range(n_epochs):

        # shuffle batches before each epoch
        np.random.shuffle(datapoints)

        # form batches
        batches = np.array_split(datapoints, n_batches)

        for n_batch, batch in enumerate(batches):

            loss_sum = 0

            # iterate over all datapoints in this batch
            for x, y in batch:
                # forward
                y_hat = forward(x, w)
                print(f"y_hat.shape={y_hat.shape}")

                loss_sum += loss(y, y_hat)

            print(f"epoch={epoch}, n_batch={n_batch}, loss_sum={loss_sum}")

            # backward
            backward(w, lr, batch)

if __name__ == "__main__":
    train()