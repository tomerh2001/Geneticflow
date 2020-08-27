#!/usr/bin/env python
# coding: utf-8

# In[42]:


import tensorflow as tf
import numpy as np


# In[2]:


def get(name):
    return globals()[name]


# In[3]:


def top(genomes, parents=2):
    """
    Selection of the top genomes from the given genome pool, the top genomes are determined by their fitness score.
    
    Parameters
    -----------
    genomes: list of Genome
        The genomes to select from.
    parents: int, optional
        The number of parents to select.
    
    Returns
    --------
    out: list of Genome
    """
    return sorted(genomes, key=lambda g: g.fitness)[-parents:]

