import numpy as np
import nltk
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
        next(f) # skip first line
        for i, line in enumerate(f):
            if i == 40000:
                # only read the first 40k lines
                break
            columns = line.split(" ")
            # print(columns)
            token_embeddings[columns[0]] = np.array(columns[1:], dtype='f')
    return token_embeddings


def tokenize_sentences(first_sentences, second_sentences):

    tokenized_first_sentences = list()
    tokenized_second_sentences = list()

    for first_sentence, second_sentence in zip(first_sentences, second_sentences):
        tokenized_first_sentences.append(nltk.word_tokenize(first_sentence))
        tokenized_second_sentences.append(nltk.word_tokenize(second_sentence))

    return tokenized_first_sentences, tokenized_second_sentences


def embed_tokens(tokens, token_embeddings):
    """
    maps all tokens to their embeddings (with np.zero fallback)
    """
    embedded_tokens = []

    for token in tokens:
        if token in token_embeddings:
            embedded_tokens.append(token_embeddings[token])
        else:
            embedded_tokens.append(np.zeros(300))

    return embedded_tokens

def create_average_sentence_embeddings(tokenized_sentences, token_embeddings):
    average_sentence_embeddings = []

    for tokenized_sentence in tokenized_sentences:
        sentence_embedding = embed_tokens(tokenized_sentence, token_embeddings)

        average_sentence_embedding = np.zeros(300)
        for token_embedding in sentence_embedding:
            average_sentence_embedding += token_embedding
        average_sentence_embedding /= len(tokenized_sentence)

        average_sentence_embeddings.append(average_sentence_embedding)

    return average_sentence_embeddings

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

    # output task 1.1
    print(first_sentences[0], second_sentences[0], scores[0])

    # task 1.2
    token_embeddings = load_token_embeddings("wiki-news-300d-1M.vec")
    tokenized_first_sentences, tokenized_second_sentences = tokenize_sentences(first_sentences, second_sentences)
    
    # output b)
    print(tokenized_first_sentences[:1])

    # output c)
    first_sentence_average_embedding = create_average_sentence_embeddings(tokenized_first_sentences[:1], token_embeddings)
    print(first_sentence_average_embedding[0][:20])
