import numpy as np
np.random.seed(7)
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import utils as np_utils
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

# read testing data from optdigits.tes in form of dataframe
testing_df = pd.read_csv('dataset/optdigits.tes', header=None)
X_testing,  y_testing = testing_df.loc[:, 0:63],  testing_df.loc[:, 64]

# read training data from optdigits.tra in form of dataframe
training_df = pd.read_csv('dataset/optdigits.tra', header=None)
X_training, y_training = training_df.loc[:, 0:63], training_df.loc[:, 64]

# scaling inputs from 0-16 to 0-1
X_training = X_training.astype('float32') / 16
X_testing = X_testing.astype('float32') / 16

# one-hot output encoding
y_training_onehot = np_utils.to_categorical(y_training)
y_testing_onehot = np_utils.to_categorical(y_testing)

num_pixels = 64
num_classes = 10

# Early stopping
callback = EarlyStopping(monitor='val_loss', patience=3)


# define model
# def create_model(hidden_layers, neurons, learning_rate, momentum,
#                  activation='tanh', loss='categorical_crossentropy'):

def create_model(hidden_layers, neurons, learning_rate, momentum,
                 activation='relu', loss='mean_squared_error'):

    # Initialize
    model = Sequential()
    # input layer
    model.add(Dense(neurons, input_dim=num_pixels, activation=activation))
    # hidden layers
    for i in range(hidden_layers):
        model.add(Dense(neurons, activation=activation))
    # output layer
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    optimizer = SGD(lr=learning_rate, momentum=momentum)

    # model.summary()

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


# Early stopping
callback = EarlyStopping(monitor='val_loss', patience=3)

# create model
my_classifier = KerasClassifier(build_fn=create_model, validation_split=0.2, verbose=0)
# define the grid search parameters
epochs = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
batches = [32, 64, 128, 256, 512]
hidden_layers = [0, 1, 2, 3]
neurons = [20, 25, 30, 35, 40, 45, 50]
learning_rate = [0.01, 0.001, 1e-3, 1e-4, 1e-5]
momentum = [0.5, 0.9, 0.99]

# Create hyperparameter options
hyperparameters = dict(epochs=epochs, batch_size=batches, hidden_layers=hidden_layers,
                       neurons=neurons, learning_rate=learning_rate, momentum=momentum)
# Create randomized search
grid = RandomizedSearchCV(estimator=my_classifier, param_distributions=hyperparameters)
# Create grid search
# grid = GridSearchCV(estimator=my_classifier, param_grid=hyperparameters)

# Fit grid search
grid_result = grid.fit(X_training, y_training_onehot, callbacks=[callback])

# View hyperparameters of best neural network
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
