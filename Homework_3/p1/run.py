import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
nltk.download('punkt')


# ------------------------------------------------
#             1.1 Data Formats
# ------------------------------------------------


def data_reader(filename):
    """
    Opens and splits datafile into three lists

    Arguments
    ---------
    filename: name of the file to read
    """
    scores = []
    first_sentences = []
    second_sentences = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            splitlist = line.strip().split("\t")
            scores.append(float(splitlist[0]))
            first_sentences.append(splitlist[1])
            second_sentences.append(splitlist[2])

    return scores, first_sentences, second_sentences


# ------------------------------------------------
#             1.2 Embedding the Sentences
# ------------------------------------------------
# Expect the embeddings to be accessible at ./wiki-news-300d-1M.vec here, but
# DO NOT upload them to Moodle!


def load_token_embeddings(filename):
    token_embeddings = dict()
    with open(filename, encoding="utf-8") as f:
        next(f) # skip first line
        for i, line in enumerate(f):
            if i == 40000:
                # only read the first 40k lines
                break
            columns = line.split(" ")
            # print(columns)
            token_embeddings[columns[0]] = np.array(columns[1:], dtype='f')
    return token_embeddings


def tokenize_sentences(sentences):

    tokenized_sentences = list()

    for sentence in sentences:
        #print(sentence)
        tokenized_sentences.append(nltk.word_tokenize(sentence))

    return tokenized_sentences


def embed_sentence(sentence, token_embeddings):
    """
    maps all tokens in sentence to their embeddings (with np.zero fallback)
    """
    embedded_sentence = []

    for token in sentence:
        if token in token_embeddings:
            embedded_sentence.append(token_embeddings[token])
        else:
            embedded_sentence.append(np.zeros(300))

    return embedded_sentence

def create_average_sentence_embeddings(tokenized_sentences, token_embeddings):
    average_sentence_embeddings = []

    for tokenized_sentence in tokenized_sentences:
        embedded_sentence = embed_sentence(tokenized_sentence, token_embeddings)

        average_sentence_embedding = np.zeros(300)
        for token_embedding in embedded_sentence:
            average_sentence_embedding += token_embedding
        average_sentence_embedding /= len(tokenized_sentence)

        average_sentence_embeddings.append(average_sentence_embedding)

    return average_sentence_embeddings

# ------------------------------------------------
#             1.3 Scoring the Similarity
# ------------------------------------------------

def task_3_prepare_data(token_embeddings):

    datasets = []
    for filepath in ['data-train.txt', 'data-dev.txt', 'data-test.txt']:
        y, x_1, x_2 = data_reader(filepath)
        y = np.array(y)
        x_1 = np.array(create_average_sentence_embeddings(tokenize_sentences(x_1), token_embeddings))
        x_2 = np.array(create_average_sentence_embeddings(tokenize_sentences(x_2), token_embeddings))
        #print(f"1={x_1}, 2={x_2}")

        datasets.append((y, x_1, x_2))

    return datasets

def task_3(token_embeddings):

    train, dev, test = task_3_prepare_data(token_embeddings)

    input_1 = Input(shape=(300,))
    input_2 = Input(shape=(300,))

    x = Concatenate()([input_1, input_2])
    x = Dropout(.3)(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(.3)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_1, input_2], outputs=x)
    model.summary()

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.fit(
        x=[train[1], train[2]], 
        y=train[0], 
        batch_size=100, 
        epochs=300, 
        verbose=1
    )

    # returns (loss, mse)
    return model.evaluate(
        x=[dev[1], dev[2]],
        y=dev[0]
    )

####################################
#                                  #
#   add your implementation here   #
#                                  #
####################################


if __name__ == "__main__":

    train_data = "data-train.txt"
    scores, first_sentences, second_sentences = data_reader(train_data)

    # # output task 1.1
    print(first_sentences[0], second_sentences[0], scores[0])

    # # task 1.2
    token_embeddings = load_token_embeddings("wiki-news-300d-1M.vec")
    tokenized_first_sentences = tokenize_sentences(first_sentences)
    tokenized_second_sentences = tokenize_sentences(second_sentences)
    
    # # output b)
    print(tokenized_first_sentences[:1])

    # # output c)
    first_sentence_average_embedding = create_average_sentence_embeddings(tokenized_first_sentences[:1], token_embeddings)
    print(first_sentence_average_embedding[0][:20])

    # task 1.3
    task_3(token_embeddings)
