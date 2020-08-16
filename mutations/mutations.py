#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
sys.path.append(os.path.realpath('..'))

import tensorflow as tf
import numpy as np
from tensorflow import keras

from utils import utils


# In[10]:


def add_node(layer):
    """
    Create a clone of the layer with an additional node.
    
    Parameters
    -----------
    layer: keras.layers.Layer
        The layer to add a note to.
    
    Returns
    --------
    out: keras.layers.Layer
        A clone of the original layer with one additional node.
    """
    units = layer.units
    inbound_layer = layer.inbound_nodes[0].inbound_layers
    outbound_layer = layer.outbound_nodes[0].outbound_layer

    clone = utils.clone_layer(layer, {'units': units+1})
    outbound_layer_clone = utils.clone_layer(outbound_layer)

    x = clone(layer.input)
    x = outbound_layer_clone(clone.output)

    clone_weights = utils.connect_weights(inbound_layer, clone)
    clone.set_weights([clone_weights, clone.bias.numpy()])
    
    outbound_layer_clone_weights = utils.connect_weights(clone, outbound_layer, from_layer2=True)
    outbound_layer_clone.set_weights([outbound_layer_clone_weights, outbound_layer.bias.numpy()])

    return clone


# In[ ]:


def remove_node(layer):
    """
    Create a clone of the layer with one node less.
    
    Parameters
    -----------
    layer: keras.layers.Layer
        The layer to remove a note from.
    
    Returns
    --------
    out: keras.layers.Layer
        A clone of the original layer with one node less.
    """
    units = layer.units
    inbound_layer = layer.inbound_nodes[0].inbound_layers
    outbound_layer = layer.outbound_nodes[0].outbound_layer

    clone = utils.clone_layer(layer, {'units': units-1})
    outbound_layer_clone = utils.clone_layer(outbound_layer)

    x = clone(layer.input)
    x = outbound_layer_clone(clone.output)

    clone_weights = utils.connect_weights(inbound_layer, clone)
    clone.set_weights([clone_weights, clone.bias.numpy()])
    
    outbound_layer_clone_weights = utils.connect_weights(clone, outbound_layer, from_layer2=True)
    outbound_layer_clone.set_weights([outbound_layer_clone_weights, outbound_layer.bias.numpy()])

    return clone


# In[4]:


def add_layer(model, nodes=1, activation='relu'):
    """
    Create a clone of the model with an additional layer at the end.
    
    Parameters
    -----------
    model: keras.Model
    
    Returns
    --------
    out: keras.Model
        A clone of the model with a new layer.
    """
    layer = keras.layers.Dense(1, activation='relu')
    x = layer(model.layers[-2].output)
    
    last_layer = model.layers[-1]
    last_layer_clone = utils.clone_layer(last_layer)
    outputs = last_layer_clone(x)
    
    weights = utils.connect_weights(layer, last_layer, from_layer2=True)
    last_layer_clone.set_weights([weights, last_layer_clone.bias.numpy()])
    return keras.Model(inputs=model.input, outputs=outputs)


# In[ ]:





# In[ ]:




