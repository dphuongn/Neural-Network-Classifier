# Neural Network Classifier



## Data Set

We will use the following data set from the UC Irvine Machine Learning Repository:

* Optical Recognition of Handwritten Digits Data Set (use **optdigits.names**, **optdigits.tra** as training data, and **optdigits.tes** as test data)
(https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits).

## What we do

We will experiment with the Neural Network classifier. We will use the softmax function for the output layer and use **1-of-c** output encoding with target values such as (1, 0, 0, ...). Use the early stopping technique to decide when to stop training (For example, you may use 20% of training data in **optdigits.tra** as the validation set).

### Experiment with fully-connected feed-forward neural networks (MLPs)

We will implement the *multinomial model* (*"bag of words"* model). In the learning phase, we will estimate the required probability terms using the training data.

#### **Sum-of-squares error (aka mean-squared-error) vs. cross-entropy error function**:
Use the ReLU units for the hidden layers. For each of the two types of error functions, experiment with different values of hyper-parameters, including number of hidden layers, number of hidden units in each layer, learning rates, momentum rates, input scaling, and so on. Compare their classification accuracy and convergence speed (the number of iterations or actual time till training stops).


#### **tanh vs. ReLU hidden units**:
Use the cross-entropy error function. For each of the two types of error functions, experiment with different values of hyper-parameters.

### Experiment with convolutional networks (CNNs)
Use the cross-entropy error function, and ReLU hidden units. Repeat previous experiments, that is, experiment with different values of hyper-parameters (note CNNs may have different types of hyper-parameters, e.g. filter size).


<!-- ___ -->

# Specification

Language used: 
* Python 3.7

Additional packages used: 

* numpy 1.18.1
* pandas 1.0.1
* scikit-learn 0.22.1
* scipy 1.4.1	(comes with scikit-learn 0.22.1)
* tensorflow 2.1.0
* keras 2.3.1
* matplotlib 3.2.0    (optional)

# How to run in command line

* Make the folder dataset in the same directory of the .py file. Inside the dataset folder are **optdigits.tes**, **optdigits.tra**.

* Navigate Mac terminal (or Window prompt) to the directory containing the .py file and enter:
    ```
    $ python file_name.py
    ```

# How to change the hyper-parameters

## In MLPs.py file

| hyper paramter    |      Line     |  Notice |
|:-------------------|:-------------:|------:|
| loss function     |     38/41     | sum-of-squares error function: loss = *mean_squared_error*, cross-entropy error function: loss= *categorical_crossentropy* (and line 46 in MLPs_result.py) |
| activation hidden unit | 38       | tanh: activation = *tanh*,  ReLU: activation= *relu* (and line 54 in MLPs_result.py) |
| number of epochs  |    68   |    
| batch size        |   69     |   
| number of hidden layers |  70|
| learning rates     | 72|
| momentum rates    | 73 |
| whether or not scaling input |  23-24 |


## In CNNs.py file

| hyper paramter    |      Line     |
|-------------------|:-------------:|
| number of epochs  |    80   |    
| batch size        |   80    |   
| number of hidden layers |  49-57 |
| whether or not scaling input |  30-31 |
| filter size | 47 |
| kernel size | 47 |
| pooling size | 50 |
| strides | 50 |
| dropout rate | 52 |
| number of hidden units in Dense layer | 56-57 |
