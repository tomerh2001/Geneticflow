#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np


# In[63]:


def connect_weights(layer1, layer2, fill_method='random', from_layer2=False):
    import utils
    
    """
    Naivley create weights that match layer1 to layer2 while filling missing connections.
    
    Parameters
    -----------
    layer1: keras.layers.Layer
        A keras layer object, the weights from this object are preserved when possible.
    layer2: keras.layers.Layer
        A keras layer object, the weights from this object are discarded.
    fill_method: str, optional
        The method used to fill missing connections, can be random/zeros/ones.
    from_layer2: bool, optional
        By default the weights are taken from the outbound layer of layer1 since keras stores the weights backwards (i.e. weights of layer1 are the connections to the previous layer), but if True then the weights would be taken directly from layer2.
    
    Returns
    -------
    out: np.array
        An array of weights which is adjusted to connect layer1 to layer2.
    """
    layer1_outbound_weights = layer2.kernel if from_layer2 else layer1.outbound_nodes[0].outbound_layer.kernel.numpy()
    layer1_outbound_weights_shape = np.shape(layer1_outbound_weights)
    layer1_new_weights_shape = (layer1.output.shape[-1], layer2.output.shape[-1])
    
    if fill_method.lower() == 'zeros':
        layer1_new_weights = np.zeros(layer1_new_weights_shape)
    elif fill_method.lower() == 'ones':
        layer1_new_weights = np.ones(layer1_new_weights_shape)
    elif fill_method.lower() == 'random':
        layer1_new_weights = np.random.uniform(-1, 1, layer1_new_weights_shape)
    else:
        raise Exception('Fill method can only be zeros/ones/random (got {}).'.format(fill_method))

    return utils.fill_array(layer1_new_weights, layer1_outbound_weights)


# In[12]:


def clone_layer(layer, config=None):
    """
    A naive implementation of cloning a keras.layers.Layer object.
    
    Parameters
    -----------
    layer: keras.layers.Layer
        A keras layer object.
    config: dict, optional
        A dict of key-value pairs to modify in the layer config.
    
    Returns
    -------
    out: keras.layers.Layer
        A clone of the original layer.
    """
    layer_config = layer.get_config()
    
    if config:
        layer_config = {**layer_config, **config}
    
    if type(layer) is tf.keras.layers.InputLayer:
        return tf.keras.layers.InputLayer.from_config(layer_config)
    elif type(layer) is tf.keras.layers.Dense:
        return tf.keras.layers.Dense.from_config(layer_config)
    else:
        raise Exception('Layer cloning of type {} is not supported currently. (note: this is a naive implementation and not a final release)'.format(type(layer)))


# In[4]:


def generate_network(inputs, hidden, outputs, activations='relu', name=None):
    """
    Generates a neural network based on the given parameters.

    Parameters
    -----------
    inputs: int
        The number of nodes in the first layer.
    hidden: int or list
        The number of initial hidden layers, this can also be a list where each value is the number of nodes in each layer (the default is 1).
    outputs: int
        The number of nodes in the last layer.
    activations: str or list, optional
        The activation functions for all the nodes in the network, this can also be a list where each value is an activation function for each layer (not including the input layer!).

    Returns
    --------
    out: keras.Model
    """
    inputs = x = tf.keras.Input(inputs)
    hidden = hidden if type(hidden) is list else [1 for i in range(hidden)]
    activations = activations if type(activations) is list else [activations for i in range(len(hidden) + 1)]

    if len(hidden) + 1 != len(activations):
        raise Exception('Expected {} activation functions but got {}.'.format(len(hidden) + 1, len(activations)))

    for nodes, activation in zip(hidden, activations[:-1]):
        x = tf.keras.layers.Dense(nodes, activation=activation)(x)
    outputs = tf.keras.layers.Dense(outputs, activation=activations[-1])(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


# In[8]:


def connect_layers(layers, weights_func=None, biases_func=None):
    """
    Naivly create a model from the given layers, the layers can be from diffrent models as long as the first layer is a keras.layers.InputLayer.
    
    Parameters
    -----------
    layers: list
        The layers to construct a model from, the first layer must be an InputLayer.
    weights_func: function, optional
        The function used to connect the weights from one layer to another, default is "connect_weights".
    biases_func: function, optional
        The function used to retrive the biases of a layer, by default the biases of the outbound layer (the layer to the "right") are taken.
        
    Returns
    --------
    out: keras.Model
    """
    if type(layers[0]) is not tf.keras.layers.InputLayer:
        raise Exception('First layer must be a keras.layers.InputLayer object, instead got {}'.format(type(layers[0])))
        
    if not weights_func:
        weights_func = connect_weights
    
    weights = []
    for layer1, layer2 in zip(layers, layers[1:]):
        layer_weights = weights_func(layer1, layer2)
        layer_biases = biases_func(layer1, layer2) if biases_func else layer2.bias.numpy()
        weights.extend([layer_weights, layer_biases])
        
        if type(layer1) is tf.keras.layers.InputLayer:
            inputs = x = clone_layer(layer1).input
        
        x = clone_layer(layer2)(x)
        
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.set_weights(weights)
    return model


# In[ ]:




