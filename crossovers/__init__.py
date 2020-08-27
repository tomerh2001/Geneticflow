#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import sys
sys.path.append(os.path.realpath('..'))

import tensorflow as tf
import numpy as np
from tensorflow import keras

import utils


# In[3]:


def get(name):
    return globals()[name]


# In[5]:


def linear(genomes):
    """
    Linear crossover, evenly combine the genetic information from all the given genomes.
    
    Parameters
    -----------
    genomes: list of Genomne
        A list of genomes with the same parameters.
    
    Returns
    --------
    out: Genome
    """
    from genomes import Genome
    new_props = utils.evenly_combine(genomes, select=lambda x: x.props)
    return Genome(**new_props)


# In[35]:


def defaultNN(genomes):
    """
    Default crossover function for the NEAT algorithm. Splits the given models evenly and prdocues a new neural network with layers from all the given genomes.

    Parameters
    -----------
    genomes: tuple
        A tuple containing 2 or more neural network models.

    Returns
    --------
    out: keras.Model
    """
    max_layers_count = max(len(genome.layers)-2 for genome in genomes)
    layers_per_genome = int(np.ceil(max_layers_count / len(models)))

    layers = [genomes[0].layers[0]]

    for i, genome in enumerate(genomes):
        start_layer_index = layers_per_genome*i
        end_layer_index = start_layer_index + layers_per_genome

        layers.extend(genome.layers[1:-1][start_layer_index:end_layer_index])

    layers.append(genomes[-1].layers[-1])
    return utils.connect_layers(layers)


# In[ ]:





# In[ ]:




