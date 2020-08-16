#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow import keras

from utils import utils
from crossovers import crossovers
from mutations import mutations


# In[2]:


class Neat:
    def __init__(self):
        self.compiled = False
    
    def compile(self, inputs, outputs, hidden=0, activations='relu', node_mutate_rate=0.01, layer_mutate_rate=0.01, bias_mutate_rate=0.7):
        """
        Compiles the neat object with network parameters.
        
        Parameters
        -----------
        inputs: int
            The number of nodes in the first layer.
        outputs: int
            The number of nodes in the last layer.
        hidden: int or list, optional
            The number of initial hidden layers, this can also be a list where each value is the number of nodes in each layer (the default is 1).
        activations: str or list, optional
            The activation functions for all the nodes in the network, this can also be a list where each value is an activation function for each layer (not including the input layer!).
        node_mutate_rate: int, optional
            The chance of a new node being added for each layer.
        layer_mutate_rate: int, optional
            The chance of a new layer being added.
        bias_mutate_rate: int, optional
            The chance of changing the bias for each layer.
            
        Returns
        --------
        out: None
        """
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = hidden
        self.activations = activations
        self.node_mutate_rate = node_mutate_rate
        self.layer_mutate_rate = layer_mutate_rate
        self.bias_mutate_rate = bias_mutate_rate
        self.compiled = True
        
    def fit(self, fit_func, population=100, generations=50, crossover=None, mutation=None):
        if not self.compiled:
            raise Exception('Neat object is not compiled, you must compile the object before training (see ".compile").')


# In[3]:


neat = Neat()
neat.compile(inputs=4, outputs=4)


# In[4]:


# history = neat.fit(func, population=100, generations=50)
# winner = neat.winner


# In[ ]:





# In[5]:


node_add_rate=0.3
node_remove_rate=0.1
layer_add_rate=0.01
layer_remove_rate=0.01
weights_change_rate=0.2
biases_change_rate=0.7


# In[6]:


model = utils.generate_network(4, [2], 4)
layer1 = model.layers[1]
layer2 = model.layers[2]

utils.connect_weights(layer1, layer2)


# In[ ]:





# In[ ]:




