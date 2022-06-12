import numpy as np
np.random.seed(42)

import os
import math
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras import metrics
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback, ModelCheckpoint

#import some additional packages
from keras.layers import Dropout, LSTM, Bidirectional, Dense
from keras.layers.wrappers import TimeDistributed

from sklearn.metrics import f1_score

from util import utils

def run(params):
    batch_size = params["batch_size"]
    model_path = params["model_path"]
    predict_file = params["predict_file"]

    # Load data: [[token11, token12, ...],[token21,token22,...]]
    # and label: [[label11, label12, ...],[label21,label22,...]]
    X_train_data, y_train_data = utils.get_task_data("pos", utils.load_dataset("ner_eng_bio.train"))
    X_dev_data, y_dev_data = utils.get_task_data("pos", utils.load_dataset("ner_eng_bio.dev"))
    X_test_data, y_test_data = utils.get_task_data("pos", utils.load_dataset("ner_eng_bio.test"))

    # Load pretrained embeddings and get reverse index
    mat_embedding, word_to_index = utils.get_pretrained_embeddings(params["embeddings"])
    index_to_word = dict((v,k) for k,v in word_to_index.items())

    # Get index -> label and label -> index dictionaries
    index_to_label = utils.get_index_dict(y_train_data + y_dev_data + y_test_data)
    label_to_index = dict((v,k) for k,v in index_to_label.items())

    # Get indexed data and labels - 1 is the OOV index
    X_train_index = [[word_to_index.get(word.lower(), 1) for word in sentence] for sentence in X_train_data]
    X_dev_index = [[word_to_index.get(word.lower(), 1) for word in sentence] for sentence in X_dev_data]
    X_test_index = [[word_to_index.get(word.lower(), 1) for word in sentence] for sentence in X_test_data]

    y_train_index = [[label_to_index[label] for label in sentence] for sentence in y_train_data]
    y_dev_index = [[label_to_index[label] for label in sentence] for sentence in y_dev_data]
    y_test_index = [[label_to_index[label] for label in sentence] for sentence in y_test_data]


    # For batch training:
    # Pad additional 0 elements at the end for the last batch:
    X_train_padded = X_train_index + [[0] for _ in range(math.ceil(len(X_train_index)/batch_size)*batch_size-len(X_train_index))]
    X_dev_padded = X_dev_index + [[0] for _ in range(math.ceil(len(X_dev_index)/batch_size)*batch_size-len(X_dev_index))]
    X_test_padded = X_test_index + [[0] for _ in range(math.ceil(len(X_test_index)/batch_size)*batch_size-len(X_test_index))]

    y_train_padded = y_train_index + [[0] for _ in range(math.ceil(len(y_train_index)/batch_size)*batch_size-len(y_train_index))]
    y_dev_padded = y_dev_index + [[0] for _ in range(math.ceil(len(y_dev_index)/batch_size)*batch_size-len(y_dev_index))]

    # Get maximum sentence length to pad instances
    max_sentence_length = max(map(lambda x: len(x), X_train_data + X_dev_data))

    # Get the number of classes:
    number_of_classes = len(index_to_label.items())

    # Pad for all inputs
    X_train = sequence.pad_sequences(X_train_padded, maxlen=max_sentence_length)
    X_dev = sequence.pad_sequences(X_dev_padded, maxlen=max_sentence_length, padding='post')
    X_test = sequence.pad_sequences(X_test_padded, maxlen=max_sentence_length, padding='post')

    # For categorical cross_entropy we need matrices representing the classes:
    # Note that we pad after doing the transformation into the matrix!
    y_train = sequence.pad_sequences(np.asarray([np_utils.to_categorical(y_label,number_of_classes+1) for y_label in y_train_padded]), maxlen=max_sentence_length)
    y_dev = sequence.pad_sequences(np.asarray([np_utils.to_categorical(y_label,number_of_classes+1) for y_label in y_dev_padded]), maxlen=max_sentence_length, padding='post')
    # We do not need to pad y_test, since we can just use y_test_index


    # ------------------------------------------------
    #             1.2 Bi-LSTM Model
    # ------------------------------------------------

    def build_model(embedding_matrix, max_sentence_length, number_of_classes):
        dropout = params["dropout"]
        n_neurons = params["hidden_units"]
        model = Sequential()
        model.add(Embedding(len(embedding_matrix),
                            len(embedding_matrix[0]),
                            input_length=max_sentence_length,
                            weights=[embedding_matrix],
                            mask_zero=True,
                            trainable=False,
                            batch_input_shape=(batch_size, max_sentence_length)))
        if params["model"] == "lstm":
            model.add(Dropout(dropout))
            model.add(LSTM(n_neurons, input_shape=(max_sentence_length, 1), return_sequences=True))
            model.add(Dropout(dropout))
            model.add(TimeDistributed(Dense(number_of_classes + 1, activation='softmax')))
        elif params["model"] == "bilstm":
            model.add(Dropout(dropout))
            model.add(Bidirectional(LSTM(n_neurons, return_sequences=True), input_shape=(max_sentence_length, 1)))
            model.add(Dropout(dropout))
            model.add(TimeDistributed(Dense(number_of_classes + 1, activation='softmax')))
        else:
            raise Exception('Unknown model!')

        model.compile('adagrad', 'categorical_crossentropy', metrics=[metrics.categorical_accuracy])
        return model

    # ------------------------------------------------
    #            1.4 F1 Model Checkpointer
    # ------------------------------------------------

    class F1ScoreModelCheckpointer(Callback):
        def __init__(self, model):
            super(F1ScoreModelCheckpointer).__init__()
            self.model = model

        def on_train_begin(self, logs={}):
            self.best_f1 = 0.0

        def on_epoch_end(self, batch, logs={}):

            # 1.: get predictions
            y_hat_dev_padded = self.model.predict(X_dev, batch_size=batch_size)

            # 2: flatten all outputs and remove padding
            y_indices_flattened = []
            y_hat_indices_flattened = []

            for sentence_y, sentence_y_hat in zip(y_dev_padded, y_hat_dev_padded):
                for token_y_index, token_y_hat_probabilities in zip(sentence_y, sentence_y_hat):

                    # check if we encounter padding
                    if token_y_index == 0:
                        # continue with next sentence
                        break

                    token_y_hat_index = token_y_hat_probabilities.tolist().index(max(token_y_hat_probabilities))

                    y_indices_flattened.append(index_to_label[token_y_index])
                    y_hat_indices_flattened.append(index_to_label[token_y_hat_index])

                    #print(f"y={token_y_index}, y_hat={token_y_hat_index}")
                    

            #print(len(y_indices_flattened), y_indices_flattened[0], len(y_hat_indices_flattened), y_hat_indices_flattened[0])

            # 3.: compupte f1 score
            #print(f"the following pos tags were not predicted: {set(y_indices_flattened) - set(y_hat_indices_flattened)}")
            self.current_f1 = f1_score(y_indices_flattened, y_hat_indices_flattened, list(index_to_label.values()), average='macro')

            # 4.: store best model
            if self.best_f1 < self.current_f1:
                # best is smaller than current -> update best and save model
                self.best_f1 = self.current_f1

                print(f"### Found new best_f1={self.best_f1}")
                model.save(params['model_path'])


            ####################################
            #                                  #
            #   add your implementation here   #
            #                                  #
            # 1. Get predictions               #
            # 2. Flatten all outputs and       #
            #    remove padding                #
            # 3. Compute f1 score              #
            # 4. Store best model              #
            #                                  #
            ####################################

            print(" F1 score %f" % (self.current_f1))
            # exit()
            return


    def train(model):
        if params["checkpointer"] == "f1":
            checkpointer = F1ScoreModelCheckpointer(model)
        elif params["checkpointer"] == "acc":
            checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True,
                                       monitor='val_categorical_accuracy', mode='max')
        else:
            raise ValueError('Unknown checkpointer "{}"'.format(params["checkpointer"]))

        model.fit(X_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=params["epochs"],
                  validation_data=(X_dev,y_dev),
                  callbacks=[checkpointer],
                  shuffle=False)

    def predict(model):
        model.load_weights(model_path)

        # Get class probabilities for the test set:
        predictions = model.predict(X_test, batch_size=batch_size)

        # Compute F1 score for test set:
        test_pred = []
        test_truth = []
        for sent_pred, sent_truth in zip(predictions, y_test_index):
            for lab, word_pred in zip(sent_truth, sent_pred):
                test_truth.append(index_to_label[lab])
                test_pred.append(index_to_label[word_pred.tolist().index(max(word_pred))])
                if word_pred.tolist().index(max(word_pred)) == 0:
                    print("Warning, PADDING label got predicted!")
        f1_test = f1_score(test_truth, test_pred, list(index_to_label.values()),average='macro')
        print("F1 score on test set: {}".format(f1_test))

        # Use unpadded inputs and omit padding
        # Output in conll format: line_number(int)  word(str)   predict_label(str)  true_label(str)
        prediction_result = utils.get_prediction_results(X_test_index, y_test_index, predictions, index_to_word, index_to_label)

        with open(predict_file, 'w') as file:
            for document in prediction_result:
                for line in document:
                    file.write("\t".join(line) + "\n")

    model = build_model(mat_embedding, max_sentence_length, number_of_classes)
    print('Model Summary:')
    print(model.summary())
    train(model)
    predict(model)


if __name__=='__main__':
    model_path = os.path.join("results", "bilstm.model")
    predict_file = os.path.join("results", "predictions.txt")

    params = {"model_path": model_path,
              "predict_file": predict_file,
              "model": "bilstm", # "lstm" or "bilstm"
              "checkpointer": "f1",        # "acc" or "f1"
              "batch_size": 10,
              "dropout": 0.5,
              "hidden_units": 100,
              "epochs": 10,
              "embeddings": "glove.6B.50d.txt"}
    run(params)