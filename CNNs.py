import numpy as np
np.random.seed(7)
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix

import matplotlib.pyplot as plt

import timeit
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD

from tensorflow.keras import utils as np_utils
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

testing_df = pd.read_csv('dataset/optdigits.tes', header=None)
X_testing,  y_testing = testing_df.loc[:, 0:63],  testing_df.loc[:, 64]

training_df = pd.read_csv('dataset/optdigits.tra', header=None)
X_training, y_training = training_df.loc[:, 0:63], training_df.loc[:, 64]


# reshape inputs to be [samples][width][height][channels]
X_training = X_training.values.reshape((X_training.shape[0], 8, 8, 1)).astype('float32')
X_testing = X_testing.values.reshape((X_testing.shape[0], 8, 8, 1)).astype('float32')

# scaling inputs from 0-16 to 0-1
X_training = X_training / 16
X_testing = X_testing / 16

# one hot encode outputs
y_training_onehot = np_utils.to_categorical(y_training)
y_testing_onehot = np_utils.to_categorical(y_testing)

print(y_training.shape)

callback = EarlyStopping(monitor='val_loss', patience=3)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    # input layer
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), input_shape=(8, 8, 1),
                     activation='relu', data_format='channels_last'))
    # MaxPooling2D layer is a way to reduce the number of parameters in our model
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format='channels_last'))
    # dropout helps protect the model from memorizing or over-fitting training data
    model.add(Dropout(0.2))
    # flatten layer turns all data into a 1D vector
    model.add(Flatten())
    # fully connected Dense layer
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # output layer
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Fit the model
estimator = KerasClassifier(build_fn=baseline_model)


# start the timer for training the neural network
start = timeit.default_timer()

# testing

# build the model
model = baseline_model()

# Fit the model
history = model.fit(X_training, y_training_onehot, validation_split=0.2,
          epochs=10, batch_size=32, verbose=0, callbacks=[callback])

# stop the timer after the neural network has been trained and tested
stop = timeit.default_timer()

# Final evaluation of the model
# training data
y_training_pred = model.predict_classes(X_training)
scores_training = model.evaluate(X_training, y_training_onehot, verbose=0)
print("\nOverall classification accuracy of the of CNNs on training data:  %.2f" % (scores_training[1]*100))
print('\nEach class accuracy of CNNs on training data:')
for c in range(10):
    print("Class", c, ":", recall_score(y_training,  y_training_pred, average=None)[c])
print('\nConfusion matrix of of CNNs on training data:\n', confusion_matrix(y_training, y_training_pred))

# testing data
y_testing_pred = model.predict_classes(X_testing, verbose=0)
scores_testing = model.evaluate(X_testing, y_testing_onehot, verbose=0)
print("\nOverall classification accuracy of CNNs on testing data: %.2f%%" % (scores_testing[1]*100))
print('\nEach class accuracy of CNNs on testing data:')
for c in range(10):
    print("Class", c, ":", recall_score(y_testing,  y_testing_pred, average=None)[c])

print('\nConfusion matrix of of CNNs on testing data:\n',
      confusion_matrix(y_testing, y_testing_pred))


print('\nTotal time of training CNNs: ', stop - start, "second")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plot epoch vs accuracy
plt.show()
