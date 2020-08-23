#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import sys
sys.path.append(os.path.realpath('..'))

import tensorflow as tf
import numpy as np


# In[ ]:





# In[30]:


class Base:
    """
    The base class is used for solving general-purpose genetic problems, it is also used as the base class for all the other genetic algorithms that are found within the `algorithms` submodule.
    """
    def __init__(self):
        self.compiled = False
    
    def compile_base(self, base_genome):
        from genomes import Genome 
        """
        Compiles the base object with evolution parameters.
        
        Parameters
        -----------
        base_genome: Genome or dict
            A base genome used to initiate the first population, if a dict is passed then a Genome would be created with the dict parameters.
            
        Returns
        --------
        out: None
        """
        self.base_genome = Genome(base_genome) if type(base_genome) is dict else base_genome
        self.compiled = True
        
    def fit(self, fit_func, population=100, generations=50, crossover=None, mutation=None, selection=None, fit_mode='multi'):
        """
        Using a basic genetic algorithm to search for the best genome for a specific problem.
        The problem is defined by the fitness function (fit_func), it is used to determine the "goal" of the process.
        
        The process of generations is comprised from the following steps: population, evaluation, selection and crossover.
        population: Creating clones of a genome each with a diffrent mutation.
        evalutation: Evaluating a genome, this is determined by the `genome.fitness` parameter of the genome returned from the `fit_func`.
        selection: The function used to select which genomes would continue on to breed.
        crossover: The process of breeding multiple genomes into a single new genome with possible new mutations.
        
        After calling `.fit` you can access the winning genome with `base.winner`.
        
        Parameters
        -----------
        fit_func: func [signature is determined by `fit_mode`]
            The fitness function used to determine the fitness of each genome, the `.fitness` parameter of the genome needs to modified within this function.
        population: int, optional
            The number of genomes to generate each generation.
        generations: int, optional
            The number of generations to run the program for.
        crossover: func(genomes: list[Genome]) -> Genome, optional
            The function used to determine how to breed the genomes, the function must take a list of genomes and return a single genome.
        mutation: func(genome: Genome) -> Genome, optional
            The function used to mutate a genome, the function must take a genome and return a mutated version of that genome.
        selection: int or func(genomes: list[Genome]) -> list[Genome], optional
            The function used to select which genomes would be passed to the `crossover` function.
        fit_mode: str, optional
            Expected values are: "single"/"multi".
            multi: Indicates to pass a list of all the population genomes into `fit_func`, this means the signature of the function must be: func(genomes: list[Genome]) -> list[Genome]
            single: Indicates to pass each genome individually to the `fit_func`, this means the signature of the function must be: func(genome: Genome) -> Genome
            
        Returns
        --------
        out: list
            The history of the evolution process.
        """
        base_genome = self.base_genome
        
        for generation in generations:
            population = self.base_genome.generate_population(population=population, mutation=mutation)
            
            if fit_mode == 'single':
                population = [fit_func(genome) for genome in population]
            elif fit_mode == 'multi':
                population = fit_func(genome)
            
            population = selection(population)
            parents = population[-self.parents:]
            
            base_genome = crossover(parents)
        
        return base_genome


# In[37]:





# In[3]:


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


# In[ ]:




