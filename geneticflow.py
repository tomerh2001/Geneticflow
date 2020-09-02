#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow import keras

import utils
import tfutils
import crossovers
import genomes
import algorithms
import selections
import mutations

from genomes import Genome # GenomeModel, GenomeLayer, 
from algorithms import Base, Neat


# In[2]:


# neat = Neat()
# neat.compile(inputs=4, hidden=2, outputs=4)
# history = neat.fit(func, population=100, generations=50)
# winner = neat.winner


# In[10]:


base_genome = Genome(a=1)

def fit_func(genome):
    genome.fitness += 2
    return genome

def mutation(genome):
    return genome

def crossover(genomes):
    return genomes[0]

def selection(genomes):
    return genomes[:3]

base = Base()
base.compile_base(base_genome)
history = base.fit(fit_func, crossover=crossover, mutation=mutation, selection=selection, verbose=1, fitness_threshold=5, fit_mode='single')


# In[ ]:




