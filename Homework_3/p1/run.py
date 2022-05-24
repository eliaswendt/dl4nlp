import numpy as np
from nltk.tokenize import word_tokenize

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
    first_sen = []
    second_sen = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            splitlist = line.strip().split("\t")

            scores.append(splitlist[0])
            first_sen.append(splitlist[1])
            second_sen.append(splitlist[2])

    return scores, first_sen, second_sen


# ------------------------------------------------
#             1.2 Embedding the Sentences
# ------------------------------------------------
# Expect the embeddings to be accessible at ./wiki-news-300d-1M.vec here, but
# DO NOT upload them to Moodle!


def load_token_embeddings(filename):
    token_embeddings = dict()
    with open(filename, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 40000:
                # only read the first 40k lines
                break
            columns = line.split("\t")
            # print(columns)
            token_embeddings[columns[0]] = np.array(columns[-1:])
    return token_embeddings


def tokenize_sentences(first_sentences, second_sentences):

    tokenized_first_sentences = list()
    tokenized_second_sentences = list()

    for first_sentence, second_sentence in zip(first_sentences, second_sentences):
        tokenized_first_sentences.append(word_tokenize(first_sentence))
        tokenized_second_sentences.append(word_tokenize(second_sentence))

    return tokenized_first_sentences, tokenized_second_sentences


def embed_tokens(tokenized_sentences, token_embeddings):
    embedded_sentences = []

    for tokenized_sentence in tokenized_sentences:
        embedded_tokens = list()
        for token in tokenized_sentence:

            if token in token_embeddings:
                embedded_sentences.append(token_embeddings[token])
            else:
                embedded_sentences.append(np.zeros(300))

        embedded_sentences.append(embedded_tokens)

    return embedded_sentences


# ------------------------------------------------
#             1.3 Scoring the Similarity
# ------------------------------------------------

####################################
#                                  #
#   add your implementation here   #
#                                  #
####################################


if __name__ == "__main__":

    train_data = "data-train.txt"
    scores, first_sentences, second_sentences = data_reader(train_data)

    # # task 1.1
    # print(first_sentence[0], second_sentence[0], scores[0])

    # task 1.2
    token_embeddings = load_token_embeddings("wiki-news-300d-1M.vec")
    tokenized_first_sentences, tokenized_second_sentences = tokenize_sentences(
        first_sentences, second_sentences, token_embeddings
    )
    print(tokenized_first_sentences[:1])
