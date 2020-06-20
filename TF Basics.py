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
import tensorflow as tf

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

To evaluate a tesnor use a session:

with tf.Session() as sess: creates a session using the default graph
    tensor_name.eval() 
"""

t = tf.zeros(shape=(2,2))
t1 = tf.ones(shape=(1,4))
