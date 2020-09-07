#!/usr/bin/env python
# coding: utf-8

# In[42]:


import tensorflow as tf
import numpy as np


# In[10]:


def get(name):
    """
    Returns a selection by name, this can be a function or a class name, in cases of a class name then an instance of that class will be returned.
    
    Parameters
    -----------
    name: str
        The name of the function or class.
    
    Returns
    --------
    out: function or instance of `Selection`.
    """
    import inspect
    target = globals()[name]
    return target() if inspect.isclass(target) else target


# In[3]:


class Selection:
    """    
    This is the class from which all selections inherit from.
    
    A selection is a callable object (much like a function) which takes a list of `Genome` objects as an input and returns a filtered list of `Genome`.
    
    The selection function of the `Selection` class is defined within the `call` function.
    
    Users will just instantiate an instance and then treat it as a callable (function).
    
    Selection is the stage of a genetic algorithm in which individual genomes are chosen from a population for later breeding (using the crossover operator).
    """
    def call(self, genomes):
        raise Exception('You are trying to invoke the default selection function of the `Selection` class, this function is not defined.')
    
    def __call__(self, genomes):
        """
        This function takes a list of genomes as an input, and returns a filtered list of `Genome`.
        
        Parameters
        -----------
        genomes: list[Genome]
            A list of genomes.
        
        Returns
        --------
        out: list[Genome]
            A filtered list of genomes.
        """
        return self.call(genomes)


# In[11]:


class Top(Selection):
    """
    Selection of the top genomes from the given genome pool, the top genomes are determined by their fitness score.
    """
    def __init__(self, parents=2):
        """
        Parameters
        -----------
        parents: int
            The number of parents to select.
        """
        self.parents = 2
        
    def call(self, genomes):
        return sorted(genomes, key=lambda g: g.fitness)[-self.parents:]

