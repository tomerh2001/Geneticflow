#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
from tensorflow import keras


# In[1]:


def connect_weights(layer1, layer2, fill_method='random'):
    """
    Naivley adjust layer1 weights to match layer2 while filling missing connections, the result is returned.
    
    Parameters
    -----------
    layer1: keras.layers.Layer
        A keras layer object, the weights from this object are preserved when possible.
    layer2: keras.layers.Layer
        A keras layer object, the weights from this object are discarded.
    fill_method: str, optional
        The method used to fill missing connections, can be random/zeros/ones.
    
    Returns
    -------
    out: np.array
        An array of weights which is adjusted to connect layer1 to layer2.
    """
    layer1_outbound_weights, layer1_outbound_biases = layer1.outbound_nodes[0].outbound_layer.get_weights()
    layer2_weights, layer2_biases = layer2.get_weights()

    layer1_outbound_weights_shape = x1, x2 = np.shape(layer1_outbound_weights)
    layer2_weights_shape = x3, x4 = np.shape(layer2_weights)
    
    if fill_method.lower() == 'zeros':
        layer1_new_weights = np.zeros(layer2_weights_shape)
    elif fill_method.lower() == 'ones':
        layer1_new_weights = np.ones(layer2_weights_shape)
    elif fill_method.lower() == 'random':
        layer1_new_weights = np.random.uniform(-1, 1, layer2_weights_shape)
    else:
        raise Exception('Fill method can only be zeros/ones/random (got {}).'.format(fill_method))
    
    layer1_new_weights[:x1, :x2] = layer1_outbound_weights[:x3, :x4]
    return layer1_new_weights


# In[ ]:


def clone_layer(layer):
    """A naive implementation of cloning a keras.layers.Layer object."""
    layer_config = layer.get_config()
    
    if type(layer) is keras.layers.InputLayer:
        batch_size, input_shape = layer.get_config()['batch_input_shape']
        return keras.Input(input_shape, batch_size)
    elif type(layer) is keras.layers.Dense:
        layer_config = layer.get_config()
        units, activation = layer_config['units'], layer_config['activation']
        return keras.layers.Dense(units, activation)
    else:
        raise Exception('Layer cloning of type {} is not supported currently. (note: this is a naive implementation and not a final release)'.format(type(layer)))

