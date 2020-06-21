from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tensorflow import feature_column as fc
"""
Learning Tensor Flow 2.0 Basics 

Tensor Flow is a Python module created by Google that can do machine learning, 
artificial intelligence and scientific computing. 

Machine Learning is subset of AI. ML attempts to take the data and its output (Label) 
to figure out a set of rules (a program). The goal is increase accuracy of models.
Below is a general schematic

Data --------->  ____________________
                |                    |
                |     ML Program     |    -----------> Rules  
                |____________________|                   |
OutPut ------->                                          |
                                                         |
                                                         |
                                                         V  
                                                        + Parameters
                                                        w/ traditional programming
                                                        |
                                                        |
                                                        |
                                                        V
                                                        Result


Deep Learning is a subset of Machine Learning and uses networking 
(a multi - layered process) as a another way to represent data 

How does TensorFlow actually works?
    
    Create graphs based on our code. Does not evualate. Like writing down the equation
    Sessions are ways to execute a part or the entire graph. Starts from the bottom of graph


What is a Tensor?
    A general vector or matrix representing higer dimensions
    Can include many data types such as float32, int32, strings etc.

""" 

# Scaler values, rank 0
string = tf.Variable("A String", tf.string)
number = tf. Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

# rank is a degree of a tensor. It is the number of dimensions involved in the tensor
rank1_tensor = tf.Variable(["Test", "Hello", "There"], tf.string)
rank2_tensor = tf.Variable([["Test", "Hello", "Bye"], ["test", "yes", "?"]], tf.string)

tensor1 = tf.ones([1,2,3]) # 2 by 3 tensor  in a nested list
tensor2 = tf.reshape(tensor1, [2, 3, 1]) # 2 by 3 tensor of element 1
tensor3 = tf.reshape(tensor2, [3, -1]) # -1 tells tensor to calculate the size of dim.

"""
Types of tesnor:
Variable (mutable)
Constant (immutable)
Placeholder (immutable)
SparseTensor (immutable)

To evaluate a tensor use a session:

with tf.Session() as sess: creates a session using the default graph
    tensor_name.eval() 


Four fundamental ML algo:
Linear Regression - used to predict numerical values for n dimensions
    STEPS for Linear Regression
    Load the data
    Explore the data
    Create a feature column to feed the model by convering categorical to numerical values
    Create an input function and a eval function
    Create the model
    Train the model using training data
    Model can now predict

Clustering
Classification
Hidden Markov Models
"""

t = tf.zeros(shape=(2,2))
t1 = tf.ones(shape=(1,4))

# titanic data set. Predict if people will survive the titanic using Linear Regression
# supverised learning example


dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv") # training data
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv") # testing data
y_train = dftrain.pop("survived") # survived in our label. Removing survived column from csv file
y_eval = dfeval.pop("survived") # Removing survived column from csv file

# print(dftrain.head()) # prints out the first five entries
# print(dftrain.describe()) # basics statistics such as mean, min, max, IQR
# print(dftrain.shape) # 627 by 9 features

"""
Data suggests that majority of passengers are in their 20's or 30's
Majority of passengers are male
Majority of passengers are in Third class
Females have a higher chance of survival

We have two data sets. Training set are usually larger than the testing set.
The purpose of training set is to develop and learn (not memorize)
Testing set is to evaulate the model and see how it is performing
Having two data sets allow our model to learn and not memorize

Convert categorical data into integers
"""
CATEGORICAL_COL = ["sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"]
NUMERIC_COL = ["age", "fare"]

features_col = []
for feature_name in CATEGORICAL_COL:
    vocab = dftrain[feature_name].unique() # output a list of uniques values of feature_name
    features_col.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))
    # above creates a column with feature_name and its unique values

for feature_name in NUMERIC_COL:
    features_col.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float64))

"""
Create an input function

Create the model
    Feed the model in batches to increase run-time and avoid crashes
    Feed the model multiple times according to the number of epochs (# of times a model sees the entire dataset)
    Ex: Epoch of 10 means the model will see the same dataset 10 times
    Start with low amount of epochs to avoid the model memorizing the dataset\
    
 

"""

# returns a function object of the input function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        if shuffle: ds = ds.shuffle(1000) # randomize data
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds # returns a batch of dataset
    return input_function #returns a function object

train_input_fn = make_input_fn(dftrain, y_train) # we feed this to the model 
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=features_col) # this is the model

# training the model
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn) # output of training model

# print(result["accuracy"]) # prints out the accuracy of the model changes base on epochs

result = list(linear_est.predict(eval_input_fn))

print(dfeval.loc[4]) # stats of the 4th person
print(y_eval.loc[4]) # did he/she survive (1 means they survive or 0 means they did not)
print(result[4]["probabilities"][1]) # chance of survival
