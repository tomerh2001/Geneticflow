#!/usr/bin/env python
# coding: utf-8

# In[56]:


import os, sys
sys.path.append(os.path.realpath('..'))

import tensorflow as tf
import numpy as np


# In[18]:


class Base:
    """
    The base class is used for solving general-purpose genetic problems, it is also used as the base class for all the other genetic algorithms that are found within the `algorithms` submodule.
    """
    def __init__(self):
        self.compiled = False
    
    def compile_base(self, base_genome):
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
        from genomes import Genome 
        self.base_genome = Genome(base_genome) if type(base_genome) is dict else base_genome
        self.compiled = True
        
    def fit(self, fit_func, population=100, generations=50, crossover='linear', mutation='MutationByType', selection=2, elitism=2, verbose=1, fitness_threshold=None, fit_mode='single'):
        """
        Using a basic genetic algorithm to search for the best genome for a specific problem.
        The problem is defined by the fitness function (fit_func) and it is used to determine the "goal" of the process.
        
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
        crossover: str or func(genomes: list[Genome]) -> Genome, optional
            The function used to determine how to breed the genomes, the function must take a list of genomes and return a single genome.
        mutation: str or func(genome: Genome) -> Genome, optional
            The function used to mutate a genome, the function must take a genome and return a mutated version of that genome.
        selection: str or int or func(genomes: list[Genome]) -> list[Genome], optional
            The function used to select which genomes would be passed to the `crossover` function, if an integer is passed then the top x (where x is the given integer) genomes with the highest fitness would be selected.
        elitism: int, optional
            Allow number of the best genomes from the current generation to carry over to the next, unaltered.
        verbose: int, optional
            The verbosity mode, expected values are 0, 1, or 2.
            0 = silent.
            1 = generation + best genome.
            2 = generation + highest fitness.
        fitness_threshold: int, optional
            A threshold for the evolution process, when the threshold has been reached the process will be terminated. 
        fit_mode: str, optional
            The fit mode, expected values are single or multi.
            multi = Indicates to pass a list of all the population genomes into `fit_func`, this means the signature of the function must be: func(genomes: list[Genome]) -> list[Genome]
            single = Indicates to pass each genome individually to the `fit_func`, this means the signature of the function must be: func(genome: Genome) -> Genome
            
        Returns
        --------
        out: list
            The history of the evolution process.
        """
        import crossovers, mutations, selections
        from genomes import Genome
        
        if type(crossover) is str:
            crossover = crossovers.get(crossover)
        if type(mutation) is str:
            mutation = mutations.get(mutation)
        if type(selection) is str:
            selection = selections.get(selection)
        elif type(selection) is int:
            parents = selection
            selection = lambda population_gens: selections.top(population_gens, parents)
        
        base_genome = self.base_genome
        elitism_gens = []
        history = []
    
        for generation in range(generations):
            population_gens = base_genome.generate_population(population=population, mutation=mutation)
            
            if elitism_gens:
                population_gens.extend(elitism_gens)
            
            if fit_mode == 'single':
                population_gens = [fit_func(genome) for genome in population_gens]
            elif fit_mode == 'multi':
                population_gens = fit_func(genome)
            else:
                raise Exception('Got unexpected fit_mode: {}.'.format(fit_mode))
            
            sorted_population_gens = sorted(population_gens, key = lambda x: x.fitness)
            
            if not isinstance(population_gens[0], Genome):
                if fit_mode == 'single':
                    raise Exception('Unexpected return type of fit_func, got {} but expected Genome.'.format(type(population_gens[0])))
                elif fit_mode == 'mutli':
                    raise Exception('Unexpected return type of fit_func, got list[{}] but expected list[Genome].'.format(type(population_gens[0])))
            
            if verbose == 1:
                print('Gen {} best genome: {}'.format(generation, sorted_population_gens[-1]))
            elif verbose == 2:
                print('Gen {} highest fitness: {}'.format(generation, max(map(lambda x: x.fitness, population_gens))))
            elif verbose != 0:
                raise Exception('Unexpected verbose value, got {} but expected values are 0/1/2'.format(verbose))

            elitism_gens = sorted_population_gens[-elitism:]
            parents_gens = selection(population_gens)
            
            if not isinstance(parents_gens, list):
                raise Exception('Unexpected return type of selection, got {} but expected list[Genome].'.format(type(parents_gens)))                
            elif not isinstance(parents_gens[0], Genome):
                raise Exception('Unexpected return type of selection, got list[{}] but expected list[Genome].'.format(type(parents_gens[0])))

            base_genome = crossover(parents_gens)
            
            if not isinstance(base_genome, Genome):
                raise Exception('Unexpected return type of crossover, got {} but expected Genome.'.format(type(base_genome)))
            
            history.append(sorted_population_gens[-1])
            
            if fitness_threshold and sorted_population_gens[-1].fitness >= fitness_threshold:
                break
        
        self.winner = sorted_population_gens[-1]
        return history


# In[ ]:


class Neat(Base):
    """
    NeuroEvolution of Augmenting Topologies (NEAT) is a genetic algorithm (GA) for the generation of evolving artificial neural networks.
    """
    def __init__(self):
        super().__init__()
    
    def compile(self, inputs, hidden, outputs, activation='relu', node_add_rate=0.1, node_remove_rate=0.1, layer_add_rate=0.01, layer_remove_rate=0.01, weights_change_rate=0.2, biases_change_rate=0.7):
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
        from genomes import GenomeModel
        base_genome = GenomeModel.generate_base(inputs, hidden, outputs, activation)
        self.compile_base(base_genome, parents, node_add_rate, node_remove_rate, layer_add_rate, layer_remove_rate, weights_change_rate, biases_change_rate)
        
    def compile_base(self, base_genome, node_add_rate=0.1, node_remove_rate=0.1, layer_add_rate=0.01, layer_remove_rate=0.01, weights_change_rate=0.2, biases_change_rate=0.7):
        """
        Compiles the neat object with a specific base genome and evolution parameters.
        
        Parameters
        -----------
        base_genome: GenomeModel
            The initial genome which is used as a base for the first population.
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
        self.node_add_rate = node_add_rate
        self.node_remove_rate = node_remove_rate
        self.layer_add_rate = layer_add_rate
        self.layer_remove_rate = layer_remove_rate
        self.weights_change_rate = weights_change_rate
        self.biases_change_rate = biases_change_rate
        super().compile_base(base_genome=base_genome)
        
    def fit(self, fit_func, population=100, generations=50, crossover=None, mutation=None, selection=None, verbose=2, fit_mode='multi'):
        """
        Using the NEAT algorithm to search for the best neural network for a specific problem.
        The problem is defined by the fitness function (fit_func) and it is used to determine the "goal" of the process.
        
        The process of generations is comprised from the following steps: population, evaluation, selection and crossover.
        population: Creating clones of a genome each with a diffrent mutation.
        evalutation: Evaluating a genome, this is determined by the `genome.fitness` parameter of the genome returned from the `fit_func`.
        selection: The function used to select which genomes would continue on to breed.
        crossover: The process of breeding multiple genomes into a single new genome with possible new mutations.
        
        After calling `.fit` you can access the winning genome with `neat.winner`.
        
        Parameters
        -----------
        fit_func: func [signature is determined by `fit_mode`]
            The fitness function used to determine the fitness of each genome, the `.fitness` parameter of the genome needs to modified within this function.
        population: int, optional
            The number of genomes to generate each generation.
        generations: int, optional
            The number of generations to run the program for.
        crossover: func(genomes: list[GenomeModel]) -> GenomeModel, optional
            The function used to determine how to breed the genomes, the function must take a list of genomes and return a single genome.
        mutation: func(genome: GenomeModel) -> GenomeModel, optional
            The function used to mutate a genome, the function must take a genome and return a mutated version of that genome.
        selection: int or func(genomes: list[GenomeModel]) -> list[GenomeModel], optional
            The function used to select which genomes would be passed to the `crossover` function, if an integer is passed then the top x (where x is the given integer) genomes with the highest fitness would be selected.
        fit_mode: str, optional
            Expected values are: "single"/"multi".
            multi: Indicates to pass a list of all the population genomes into `fit_func`, this means the signature of the function must be: func(genomes: list[GenomeModel]) -> list[GenomeModel]
            single: Indicates to pass each genome individually to the `fit_func`, this means the signature of the function must be: func(genome: GenomeModel) -> GenomeModel
            
        Returns
        --------
        out: list
            The history of the evolution process.
        """
        return super().fit(fit_func=fit_func, population=population, generations=generations, crossover=crossover, mutation=mutation, selection=selection, fit_mode=fit_mode)


# In[ ]:




