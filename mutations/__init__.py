#!/usr/bin/env python
# coding: utf-8

# In[33]:


import os, sys
sys.path.append(os.path.realpath('..'))

import tensorflow as tf
import numpy as np
import string

import utils


# In[1]:


def get(name):
    """
    Returns a mutation by name, this can be a function or a class name, in cases of a class name then an instance of that class will be returned.
    
    
    """
    import inspect
    target = globals()[name]
    return target() if inspect.isclass(target) else target


# In[5]:


class Mutation:
    """
    This is the class from which all mutations inherit from.
    
    A mutation is a callable object (much like a function) which takes a `Genome` object as an input and outputs a mutated genome.
    
    Users will just instantiate a mutation and then treat it as a callable (function).
    """
    def mutate(self, genome):
        raise Exception('You are trying to invoke the default mutation function of the `Mutation` class, this function is not defined.')
    
    def __call__(self, genome):
        return self.mutate(genome)


# In[7]:


class MutationByType(Mutation):
    """
    Invoking this mutation on a genome would apply pre-determined functions to each property (DNA) of the genome depending on their type.
    """
    
    def __init__(self, int_func=None, float_func=None, str_func=None, array_like_func=None, other_func=None):
        """
        Initializing the parameters of the Mutation By Type class.
        
        Parameters
        -----------
        int_func: func(val: any) -> any, [optional, default = mutate_int]
            The function used to mutate an integer property.
        num_func: func(val: any) -> any, [optional, default = mutate_float]
            The function used to mutate a float property.
        str_func: func(val: any) -> any, [optional, default = mutate_array_like]
            The function used to mutate a string property.
        array_func: func(val: any) -> any, [optional, default = mutate_string]
            The function used to mutate any array_like property.
        other_func: dict, optional
            A dictionary where each entry is of the structure {type: func}, each function need to be of the structure func(value: any) -> any.
        """
        self.funcs = {}
        self.funcs[int] = self.funcs[int] = int_func if int_func else mutate_int
        self.funcs[float] = self.funcs[float] = float_func if float_func else mutate_float
        self.funcs['array_like'] = array_like_func if array_like_func else mutate_array_like
        self.funcs[str] = str_func if str_func else mutate_string
        if other_func: self.funcs = {**self.funcs, **other_func}
        
    def mutate(self, genome):
        """
        Applying the given functions on each property of the genome depending on it's type.
        
        Parameters
        -----------
        genome: Genome
            The genome to mutate.
            
        Returns
        --------
        out: Genome
            A mutated clone of the original genome.
        """
        clone = genome.clone()
        
        for prop in genome.props:
            x = genome.props[prop]
            dtype = type(x)
            
            if utils.is_array_like(x):
                genome.props[prop] = self.funcs['array_like'](x)
            elif dtype in self.funcs:
                genome.props[prop] = self.funcs[dtype](x)
            else:
                raise Exception('MutationByType encountered an unexpected type ({}) which was not specified, please pass an appropriate function to the MutationByType constructor.'.format(dtype))
        
        return genome


# In[1]:


def mutate_string(text, change_rate=0.5, change_length_rate=0.3, add_rate=0.5, chars=None):
    """
    Randomly change letters from the given string, with a chance to change the text length, if text length is changed then the chance of a new char to be added is `add_rate` while the chance of a char to be removed is `1 - add_rate`.
    
    Parameters
    -----------
    text: str
        The text to mutate.
    change_rate: float (0, 1), optional
        The probability for each character in the text to change.
    change_length_rate: float (0, 1), optional
        The probability for a change in the length of the text.
    add_rate: float (0, 1), optional
        The probability for a new character to be added to the text (given the change_length_rate), this would result that the probability to remove a character from the string is `1 - add_rate`.
    chars: str, [optional, default = string.printable]
        The characters space from whom new characters are generated.
        
    Returns
    --------
    out: str
        A mutated string.
    """
    chars = chars if chars else string.printable
    for i, c in enumerate(text):
        if utils.chance(change_rate):
            text_list = [*text]
            text_list[i] = np.random.choice(list(chars))
            text = ''.join(text_list)
    
    if utils.chance(change_length_rate):
        if utils.chance(add_rate):
            text += np.random.choice(list(chars))
        else:
            text = text[:-1]
    return text


# In[152]:


def mutate_int(x):
    """
    Draws a new integer from the range (0, x*2)
    
    Parameters
    -----------
    x: int
        The integer to mutate.
    
    Returns
    --------
    out: int
        The mutated integer.
    """
    return np.random.randint(0, (x*2) + 1)


# In[151]:


def mutate_float(x):
    """
    Draws a new float from the range (0, x*2)
    
    Parameters
    -----------
    x: float
        The float to mutate.
    
    Returns
    --------
    out: float
        The mutated float.
    """
    return np.random.uniform(0, (x*2) + 1)


# In[11]:


def mutate_array_like(x):
    """
    Generates a new array with the shape of the given array using np.random.randint, the values are drawn from the range (0, max(x)).
    
    Parameters
    -----------
    x: array_like
        The array to mutate.
    
    Returns
    --------
    out: array_like
        The mutated array_like.
    """
    x_array = np.array(x)
    return np.random.randint(0, max(x_array), *x_array.shape).astype(x_array.dtype)


# In[ ]:


# def mutate_nn_genome(genome, node_add_rate=0.1, node_remove_rate=0.1, layer_add_rate=0.01, layer_remove_rate=0.01, weights_change_rate=0.2, biases_change_rate=0.7):
#     """
#     Returns a mutated version of the given genome.
#     """
#     for layer in genome.layers:
#         if chance(layer_remove_rate):
#             genome.remove_layer(layer)
#         elif chance(node_add_rate):
#             genome.add_node(layer)
#         elif chance(node_remove_rate):
#             genome.remove_node(layer)
        
#         if chance(weights_change_rate):
#             genome.change_weights(layer)
#         if chance(biases_change_rate):
#             genome.change_biases(layer)
    
#     if chance(layer_add_rate):
#         genome.add_hidden_layer()
    
#     return genome

