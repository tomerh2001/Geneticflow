#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
sys.path.append(os.path.realpath('..'))

from tensorflow import keras
import numpy as np


# In[5]:


def chance(rate, size=None):
    """
    Returns true based on the propabilty (rate) provided.
    
    Parameters
    -----------
    rate: int
        A number between 0-1 that represents a probability.
    size: int or tuple, optional
        The shape of the output.
    
    Returns
    --------
    out: bool
    """
    return np.random.random_sample(size) < rate


# In[60]:


def fill_array(a, b):
    """
    Fill an array with another array while maintaining original values where they have not been overwritten.
    
    Parameters
    -----------
    a: array or list
        An array.
    b: array or list
        An array.
    
    Returns
    --------
    out: np.array
        A copy of the first array where values from the second array are overwritten in the shared dimensions between the two arrays.
        
    Examples
    ---------
    >> a = np.zeros((3, 3))
    >> b = np.ones((2, 2))
    
    # array([[1., 1., 0.],
    #       [1., 1., 0.],
    #       [0., 0., 0.]])
    """
    a, b = np.array(a), np.array(b)
    x1, x2 = np.min((a.shape, b.shape), 0)
    c = a.copy()
    c[:x1, :x2] = b[:x1, :x2]
    return c


# In[1]:


def mutate_genome(genome, node_add_rate=0.1, node_remove_rate=0.1, layer_add_rate=0.01, layer_remove_rate=0.01, weights_change_rate=0.2, biases_change_rate=0.7):
    """
    Returns a mutated version of the given genome.
    """
    for layer in genome.layers:
        if chance(layer_remove_rate):
            genome.remove_layer(layer)
        elif chance(node_add_rate):
            genome.add_node(layer)
        elif chance(node_remove_rate):
            genome.remove_node(layer)
        
        if chance(weights_change_rate):
            genome.change_weights(layer)
        if chance(biases_change_rate):
            genome.change_biases(layer)
    
    if chance(layer_add_rate):
        genome.add_hidden_layer()
    
    return genome


# In[ ]:




