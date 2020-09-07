#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
sys.path.append(os.path.realpath('..'))

import tensorflow as tf
import numpy as np
from tensorflow import keras

import utils


# In[38]:


def get(name):
    """
    Returns a crossover by name, this can be a function or a class name, in cases of a class name then an instance of that class will be returned.
    
    Parameters
    -----------
    name: str
        The name of the function or class.
    
    Returns
    --------
    out: function or instance of `Crossover`.
    """
    import inspect
    target = globals()[name]
    return target() if inspect.isclass(target) else target


# In[31]:


class Crossover:
    """
    This is the class from which all crossovers inherit from.
    
    A crossover is a callable object (much like a function) which takes a list of `Genome` objects as an input and outputs a single `Genome` object.
    
    The crossover function of the `Crossover` class is defined within the `call` function.
    
    Users will just instantiate an instance and then treat it as a callable (function).
    """
    def call(self, genomes):
        raise Exception('You are trying to invoke the default crossover function of the `Crossover` class, this function is not defined.')
    
    def __call__(self, genomes):
        """
        This function takes a list of genomes as an input, and returns a single genome which is the "child" of the given genomes.
        
        Parameters
        -----------
        genomes: list[Genome]
            A list of genomes.
        
        Returns
        --------
        out: Genome
            A child of the given genomes.
        """
        return self.call(genomes)


# In[30]:


class EvenlyCombine(Crossover):
    """
    This crossover invokes the `utils.evenly_combine` function (found in the utils module), 
    it would take even slices from each genome props and return a new genome containing props from each of the given genomes.
    
    For more information, see `utils.evenly_combine`.
    
    Examples
    ---------
    >>> g1 = Genome(a=1, b=1, c=1)
    >>> g2 = Genome(a=2, b=2, c=2)
    >>> EvenlyCombine()([g1, g2])
    <Genome fitness="0" a="1" b="1" c="2">
    
    >>> g1 = Genome(a=1, b=1, c=1)
    >>> g2 = Genome(a=2, b=2, c=2)
    >>> g3 = Genome(a=3, b=3, c=3)
    >>> EvenlyCombine()([g1, g2, g3])
    <Genome fitness="0" a="1" b="2" c="3">
    """
    
    def __call__(self, genomes):
        from genomes import Genome
        new_props = utils.evenly_combine(genomes, select=lambda x: x.props)
        return Genome(**new_props)


# In[40]:


# def defaultNN(genomes):
#     """
#     Default crossover function for the NEAT algorithm. Splits the given models evenly and prdocues a new neural network with layers from all the given genomes.

#     Parameters
#     -----------
#     genomes: tuple
#         A tuple containing 2 or more neural network models.

#     Returns
#     --------
#     out: keras.Model
#     """
#     max_layers_count = max(len(genome.layers)-2 for genome in genomes)
#     layers_per_genome = int(np.ceil(max_layers_count / len(models)))

#     layers = [genomes[0].layers[0]]

#     for i, genome in enumerate(genomes):
#         start_layer_index = layers_per_genome*i
#         end_layer_index = start_layer_index + layers_per_genome

#         layers.extend(genome.layers[1:-1][start_layer_index:end_layer_index])

#     layers.append(genomes[-1].layers[-1])
#     return utils.connect_layers(layers)


# In[ ]:





# In[ ]:




