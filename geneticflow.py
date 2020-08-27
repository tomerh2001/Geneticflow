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

from genomes import GenomeModel, GenomeLayer, Genome
from algorithms import Base, Neat


# In[2]:


# neat = Neat()
# neat.compile(inputs=4, hidden=2, outputs=4)
# history = neat.fit(func, population=100, generations=50)
# winner = neat.winner


# In[ ]:





# In[22]:


import matplotlib.pyplot as plt

def fit_func(genome):
    genome.fitness = genome.img.min()
    return genome

def mutation(genome):
    for i in range(10):
        if utils.chance(1/3):
            genome.img[i] = np.random.randint(0, 255, 10)
    return genome

def crossover(genomes):
    new_img = np.concatenate((genomes[0].img[:5], genomes[1].img[5:10]))
    return Genome(img=new_img)
    
base_genome = Genome(img=np.random.randint(0, 255, (10, 10)))

base = Base()
base.compile_base(base_genome)
history = base.fit(fit_func, population=1000, generations=100, crossover=crossover, mutation=mutation, fit_mode='single')
plt.imshow(base.winner.img)


# In[17]:




