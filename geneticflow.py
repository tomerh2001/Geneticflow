#!/usr/bin/env python
# coding: utf-8

# In[33]:


import tensorflow as tf
import numpy as np
from tensorflow import keras

import utils
import tfutils
import crossovers
import genomes

from genomes import GenomeModel, GenomeLayer


# In[ ]:





# In[32]:


class Neat:
    def __init__(self):
        self.compiled = False
    
    def compile(self, inputs, hidden, outputs, activation='relu', parents=2, node_add_rate=0.1, node_remove_rate=0.1, layer_add_rate=0.01, layer_remove_rate=0.01, weights_change_rate=0.2, biases_change_rate=0.7):
        """
        Compiles the neat object with parameters for the base genome and the evolution process.
        Calling this function would create a base genome from the given network parameters.
        
        Parameters
        -----------
        inputs: int, optional
            The number of nodes in the input layer of the base genome.
        hidden: int, optional
            The number of initial hidden layers in the base genome, each layer starts with a single node.
        outputs: int, optional
            The number of nodes in the output layer of the base genome.
        activation: str, optional
            The activation function for all of the initial layers of the base genome.
        parents: int, optional
            The maximum number of parents required to create a child, the default is 2, meaning that a child would be created from the 2 fittest genomes.
        node_mutate_rate: int, optional
            The chance of a new node being added for each layer.
        layer_mutate_rate: int, optional
            The chance of a new layer being added.
        bias_mutate_rate: int, optional
            The chance of changing the bias for each layer.
            
        Returns
        --------
        out: None
        """
        base_genome = GenomeModel.generate_base(inputs, hidden, outputs, activation)
        self.compile_base(base_genome, parents, node_add_rate, node_remove_rate, layer_add_rate, layer_remove_rate, weights_change_rate, biases_change_rate)
        
    def compile_base(self, base_genome, parents=2, node_add_rate=0.1, node_remove_rate=0.1, layer_add_rate=0.01, layer_remove_rate=0.01, weights_change_rate=0.2, biases_change_rate=0.7):
        """
        Compiles the neat object with a specific base genome and evolution parameters.
        
        Parameters
        -----------
        base_genome: GenomeModel
            The initial genome which is used as a base for the first population.
        parents: int, optional
            The maximum number of parents required to create a child, the default is 2, meaning that a child would be created from the 2 fittest genomes.
        node_mutate_rate: int, optional
            The chance of a new node being added for each layer.
        layer_mutate_rate: int, optional
            The chance of a new layer being added.
        bias_mutate_rate: int, optional
            The chance of changing the bias for each layer.
            
        Returns
        --------
        out: None
        """
        self.base_genome = base_genome
        self.parents = parents
        self.node_add_rate = node_add_rate
        self.node_remove_rate = node_remove_rate
        self.layer_add_rate = layer_add_rate
        self.layer_remove_rate = layer_remove_rate
        self.weights_change_rate = weights_change_rate
        self.biases_change_rate = biases_change_rate
        self.compiled = True
        
    def fit(self, fit_func, population=100, generations=50, crossover=None, mutation=None, fit_mode='multi'):
        base_genome = self.base_genome
        
        for generation in generations:
            population = GenomeModel.generate_population(base_genome, mutation=mutation)
            
            if fit_mode == 'single':
                population = [fit_func(genome) for genome in population]
            elif fit_mode == 'multi':
                population = fit_func(genome)
            
            population.sort(key=lambda g: g.fitness)
            parents = population[-self.parents:]
            
            base_genome = crossover(parents)
        
        return base_genome


# In[31]:


neat = Neat()
neat.compile(inputs=4, hidden=2, outputs=4)


# In[4]:


# history = neat.fit(func, population=100, generations=50)
# winner = neat.winner


# In[ ]:




