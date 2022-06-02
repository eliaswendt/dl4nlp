import numpy as np
import random
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *

# ------------------------------------------------
#             2.1 Creating Data Splits
# ------------------------------------------------

################################
input_file = 'data.txt'
################################

tmp_dir = '/tmp'
train_verbose = 1
pad_length = 300


def read_data(input_file):
    vocab = {0}
    data_x = []
    data_y = []
    with open(input_file) as f:
        for line in f:
            label, content = line.split('\t')
            content = [int(v) for v in content.split()]
            vocab.update(content)
            data_x.append(content)
            label = tuple(int(v) for v in label.split())
            data_y.append(label)

    data_x = pad_sequences(data_x, maxlen=pad_length)
    return list(zip(data_y, data_x)), vocab


data, vocab = read_data(input_file)
vocab_size = max(vocab) + 1

# random seeds
random.seed(42)
tf.random.set_seed(42)

random.shuffle(data)
input_len = len(data)

# train_y: a list of 20-component one-hot vectors representing newsgroups
# train_x: a list of 300-component vectors where each entry corresponds to a word ID
train_y, train_x = zip(*(data[:(input_len * 8) // 10]))
dev_y, dev_x = zip(*(data[(input_len * 8) // 10: (input_len * 9) // 10]))
test_y, test_x = zip(*(data[(input_len * 9) // 10:]))

# ------------------------------------------------
#                 2.2 A Basic CNN
# ------------------------------------------------

train_x, train_y = np.array(train_x), np.array(train_y)
dev_x, dev_y = np.array(dev_x), np.array(dev_y)
test_x, test_y = np.array(test_x), np.array(test_y)

# Leave those unmodified and, if requested by the task, modify them locally in the specific task
batch_size = 64
embedding_dims = 100
epochs = 2
filters = 75
kernel_size = 3  # Keras uses a different definition where a kernel size of 3 means that 3 words are convolved at each step

model = Sequential()

model.add(Embedding(vocab_size, embedding_dims, input_length=pad_length))

####################################

# add convolutional layer with 75 filters and filter size = 2, ReLU activation fxn
model.add(Conv1D(75, 2, activation='relu'))
# global max pooling layer
model.add(GlobalMaxPooling1D())
#model.add(Flatten())
# softmax output layer
model.add(Dense(20, activation='softmax'))

####################################

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=train_verbose)
print('[T 2.1]: Accuracy of simple CNN: %f\n' % model.evaluate(dev_x, dev_y, verbose=0)[1])

# print model summary 
model.summary()

# ------------------------------------------------
#                2.3 Early Stopping
# ------------------------------------------------

epochs = 50
checkpoint_filepath = 'checkpoint_best'
monitor = 'loss'


model = Sequential()
model.add(Embedding(vocab_size, embedding_dims, input_length=pad_length))

# add convolutional layer with 75 filters and filter size = 2, ReLU activation fxn
model.add(Conv1D(75, 2, activation='relu'))
# global max pooling layer
model.add(GlobalMaxPooling1D())
#model.add(Flatten())
# softmax output layer
model.add(Dense(20, activation='softmax'))


save_best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor=monitor,
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch',
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor=monitor,
    min_delta=0,
    patience=2,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False # we do this manually with save_best_checkpoint_callback and model.load()
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(dev_x, dev_y, batch_size=batch_size, epochs=epochs, verbose=train_verbose, callbacks=[save_best_checkpoint_callback, early_stopping_callback])

# load model checkpoint before eval
model.load_weights(checkpoint_filepath)
print(
    '[T 2.3]: dev acc.: {}, test acc.: {}'.format(
        model.evaluate(dev_x, dev_y, verbose=0)[1],
        model.evaluate(test_x, test_y, verbose=0)[1]
    )
)



####################################
#                                  #
#   add your implementation here   #
#                                  #
####################################