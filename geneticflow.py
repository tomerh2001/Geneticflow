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

from genomes import GenomeModel, GenomeLayer
from algorithms import Base, Neat


# In[2]:


neat = Neat()
neat.compile(inputs=4, hidden=2, outputs=4)
history = neat.fit(func, population=100, generations=50)
winner = neat.winner


# In[4]:


base = Base()
base.compile_base({'a': 0, 'b':0})
base.fit(func, population=100, generations=50, mutation=mutation, crossover=crossover)


# In[ ]:


# Todo
# Decide when to apply fill & clear.
# Decide how to create a keras model from genome.
# Rename where "Find appropriate name" comments are found.
# Add mutations module with `mutate_genome`


# In[ ]:




