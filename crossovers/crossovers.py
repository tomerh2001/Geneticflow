#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import sys
sys.path.append(os.path.realpath('..'))

import tensorflow as tf
import numpy as np
from tensorflow import keras

from utils import utils


# In[1]:


def default(models):
    """
    Default crossover function. Produces a new neural network from the given models.

    Parameters
    -----------
    models: tuple
        A tuple containing 2 neural network models.

    Returns
    --------
    out: keras.Model
    """
    max_layers_count = max(len(model.layers)-2 for model in models)
    layers_per_model = int(np.ceil(max_layers_count / len(models)))

    layers = [models[0].layers[0]]

    for i, model in enumerate(models):
        start_layer_index = layers_per_model*i
        end_layer_index = start_layer_index + layers_per_model

        layers.extend(model.layers[1:-1][start_layer_index:end_layer_index])

    layers.append(models[-1].layers[-1])
    return utils.connect_layers(layers)


# In[ ]:




