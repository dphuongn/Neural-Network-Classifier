# Neural Network Classifier



## Data Set
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. It was originally collected by Ken Lang, probably for his Newsweeder: Learning to filter netnews [1] paper, though he did not explicitly mention this collection. The 20 newsgroups collection has become a popular data set for
experiments in text applications of machine learning techniques, such as text classification and text clustering.

We will use the following data set from the UC Irvine Machine Learning Repository:

* Optical Recognition of Handwritten Digits Data Set (use **optdigits.names**, **optdigits.tra** as training data, and **optdigits.tes** as test data)
(https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits).

## What we do

We will experiment with the Neural Network classifier. You will use the softmax function for the output layer and use 1-of-c output encoding with target values such as (1, 0, 0, ...). Use the early stopping technique to decide when to stop training (For example, you may use 20% of training data in **optdigits.tra** as the validation set).

### Experiment with fully-connected feed-forward neural networks

We will implement the *multinomial model* (*"bag of words"* model). In the learning phase, you will estimate the required probability terms using the training data.

#### **Sum-of-squares error (aka mean-squared-error) vs. cross-entropy error function**:
Use the ReLU units for the hidden layers. For each of the two types of error functions, experiment with different values of hyper-parameters, including number of hidden layers, number of hidden units in each layer, learning rates, momentum rates, input scaling, and so on. Compare their classification accuracy and convergence speed (the number of iterations or actual time till training stops).


#### **tanh vs. ReLU hidden units**:
Use the cross-entropy error function. For each of the two types of error functions, experiment with different values of hyper-parameters.


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

