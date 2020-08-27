#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
sys.path.append(os.path.realpath('..'))

import tensorflow as tf
import numpy as np

import utils


# In[4]:


def get(name):
    return globals()[name]


# In[2]:


def mutate_genome(genome, rates=1/3, initializer=None):
    """
    Default genome mutation function.
    Applying mutations based on the given rates, if rates is an integer then all parameters will have a chance in x (where x is the integer) to be mutated.
    
    Parameters
    -----------
    genome: Genome
        The genome to mutate.
    rates: int or dict, optional
        The chances of each parameter to mutate, if dict is passed then each entry needs to be in the structure {param_name: rate}, otherwise if an integer is passed then all parameters will have a chance of x (where x is the integer) to be mutated.
    initalizer: func(shape) -> val
        The function used to create a new value, the function receives a shape and returns a new value, by default this function is numpy `rand` func.
    Returns
    --------
    out: Genome
        The mutated genome.
    """
    if not initializer:
        initializer = np.random.rand
       
    clone = genome.clone()
    for param in clone.props:
        rate = rates[param] if type(rates) is dict else rates
        
        if utils.chance(rate):
            shape = np.shape(clone.props[param])
            clone.props[param] = initializer(*shape)
        
    return clone


# In[3]:


def mutate_nn_genome(genome, node_add_rate=0.1, node_remove_rate=0.1, layer_add_rate=0.01, layer_remove_rate=0.01, weights_change_rate=0.2, biases_change_rate=0.7):
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




