import numpy as np
np.random.seed(7)
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix

import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import EarlyStopping
from time import time

# read testing data from optdigits.tes in form of dataframe
testing_df = pd.read_csv('dataset/optdigits.tes', header=None)
X_testing,  y_testing = testing_df.loc[:, 0:63],  testing_df.loc[:, 64]

# read training data from optdigits.tra in form of dataframe
training_df = pd.read_csv('dataset/optdigits.tra', header=None)
X_training, y_training = training_df.loc[:, 0:63], training_df.loc[:, 64]

# scaling inputs from 0-16 to 0-1
# X_training = X_training.astype('float32') / 16
# X_testing = X_testing.astype('float32') / 16

# one-hot output encoding
y_training_onehot = np_utils.to_categorical(y_training)
y_testing_onehot = np_utils.to_categorical(y_testing)

num_pixels = 64
num_classes = 10

# Early stopping
callback = EarlyStopping(monitor='val_loss', patience=3)


# define  model
def create_model(neurons, activation):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_shape=(num_pixels,), activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = SGD(lr=0.01, momentum=0.99)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model


# start timer
start = time()

# build the model
model = create_model(neurons=30, activation='relu')
model.summary()

# Fit the model
history = model.fit(X_training, y_training_onehot,
          validation_split=0.2, epochs=350, batch_size=256, callbacks=[callback], verbose=0)


# Final evaluation of the model
# training data
y_training_pred = model.predict_classes(X_training, verbose=0)

scores_training = model.evaluate(X_training, y_training_onehot, verbose=0)
print("\nOverall classification accuracy of MLPs on training data: %.2f%%" % (scores_training[1]*100))
print('\nEach class accuracy of MLPs on training data:')

for c in range(10):
    print("Class", c, ":", recall_score(y_training,  y_training_pred, average=None)[c])

print('\nConfusion matrix of of MLPs on training data:\n', confusion_matrix(y_training, y_training_pred))

# testing data
y_testing_pred = model.predict_classes(X_testing, verbose=0)

scores_testing = model.evaluate(X_testing, y_testing_onehot, verbose=0)
print("\nOverall classification accuracy of MLPs on testing data: %.2f%%" % (scores_testing[1]*100))
print('\nEach class accuracy of MLPs on testing data:')

for c in range(10):
    print("Class", c, ":", recall_score(y_testing,  y_testing_pred, average=None)[c])

print('\nConfusion matrix of of MLPs on testing data:\n', confusion_matrix(y_testing, y_testing_pred))

print('\nTotal time of training MLPs:', time()-start, 's')


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plot epoch vs accuracy
plt.show()