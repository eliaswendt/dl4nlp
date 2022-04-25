import numpy as np
#from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
#               5.1 DATASET READER
# ----------------------------------------------------------------------------------------------------------------------
def read_dataset(filepath: str):

    datapoints = []

    with open(filepath) as f:
        for line in f:
            _, sentiment_str, embedding_str = line.split("\t")

            y = np.array(1.0 if sentiment_str == "label=POS" else 0.0)

            embedding = [x for x in embedding_str.split(" ")]
            embedding.append(1.0)
            x = np.array(embedding, dtype=np.double)

            datapoints.append((x, y))

    return datapoints

# ----------------------------------------------------------------------------------------------------------------------
#               5.2 NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------------------

def sig(x):
    return 1 / (1 + np.exp(-x))


def sig_deriv(x):
    return sig(x)*(1-sig(x))


def loss(y, y_hat):
    return (y_hat - y)**2 


def square_loss(datapoints, w):
    sum = 0

    for x, y in datapoints:
        sum += (forward(x, w) - y) ** 2

    return sum


def accuracy(datapoints, w):
    tp = 0
    for x, y in datapoints:
        predicted = np.round(forward(x, w))

        if predicted == y:
            tp += 1

    return tp / len(datapoints)

# ----------------------------------------------------------------------------------------------------------------------
#               5.3 TRAINING
# ----------------------------------------------------------------------------------------------------------------------

def forward(x, w):
    return sig(x.T.dot(w)) 


def backward(w, lr, batch):

    sum = np.zeros(w.shape)

    for x, y in batch:

        # print(f"x.shape={x.shape}, y.shape={y.shape}, w.shape={w.shape}")

        sum += (forward(x, w) - y) * sig_deriv(x.T.dot(w)) * x

    # print(f"sum.shape={sum.shape}, w.shape={w.shape}")

    w -= (lr / len(batch)) * sum


def train():
    # initialize weights randomly (with seed)
    np.random.seed(100)


    datapoints_dev = read_dataset("DATA/rt-polarity.dev.vecs")
    datapoints_test = read_dataset("DATA/rt-polarity.test.vecs")
    datapoints_train = read_dataset("DATA/rt-polarity.train.vecs")

    datapoints = datapoints_train

    # organize input in batches
    # shuffle batches

    N = len(datapoints[0][0]) # length of embedding vec

    # reshaping is very important here to allow multiplication in backward()
    w = np.random.normal(0, 1, (N, 1)).reshape(101,)

    lr = .01
    n_epochs = 50
    batch_size = 10

    n_batches = np.ceil(len(datapoints) / batch_size)

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

                loss_sum += loss(y, y_hat)

            #print(f"epoch={epoch}, n_batch={n_batch}, loss_sum={loss_sum}")

            # backward
            backward(w, lr, batch)

        # print matrics
        print("epoch={}/{}\n\tsquare_loss(dev)={:.2f}\n\tsquare_loss(test)={:.2f}\n\tacc(dev)={:.2f}\n\tacc(test)={:.2f}\n".format(
            epoch+1, n_epochs,
            square_loss(datapoints_dev, w), 
            square_loss(datapoints_test, w), 
            accuracy(datapoints_dev, w),
            accuracy(datapoints_test, w)
        ))

if __name__ == "__main__":
    train()