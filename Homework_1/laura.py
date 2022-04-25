import numpy as np
import os

# ----------------------------------------------------------------------------------------------------------------------
#               5.1 DATASET READER
# ----------------------------------------------------------------------------------------------------------------------
def read_dataset(filepath):
    with open(filepath) as f:
        x = []
        y = []
        for line in f:
            review, sentiment_label, sentiment_embedding = line.split("\t")
            if sentiment_label == "label=POS":
                sentiment_label = 1
            else:
                sentiment_label = 0
            y.append([sentiment_label])
            x.append([sentiment_embedding])
        data = np.column_stack((x, y))
        data = np.transpose(data)

        # print(data.shape)

        # print first row
        print(data[:, 1])

    return data


read_dataset("hw01_data/rt-polarity.train.vecs")
# ----------------------------------------------------------------------------------------------------------------------
#               5.2 NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------------------------------------------------------
#               5.3 TRAINING
# ----------------------------------------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     train_data = read_dataset("hw01_data/rt-polarity.train.vecs")
#     dev_data = read_dataset("hw01_data/rt-polarity.dev.vecs")
#     test_data = read_dataset("hw01_data/rt-polarity.test.vecs")

