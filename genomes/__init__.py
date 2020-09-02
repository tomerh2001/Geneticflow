#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os, sys
sys.path.append(os.path.realpath('..'))

import tensorflow as tf
import numpy as np
import copy

import utils


# In[2]:


class Genome:
    """
    Represents a genome. Used to hold the "DNA" of an object.
    This is the base class for all other genomes.
    """
    def __init__(self, name=None, **props):
        self.name = name
        self.fitness = 0
        self.props = props
        
    def clone(self):
        """Returns a deep copy of the current genome."""
        return Genome(name=self.name, **copy.deepcopy(self.props))
    
    def generate_population(self, population, mutation):
        """
        Generates a population of the current genome, each with diffrent mutations.
        
        Parameters
        -----------
        population: int
            The number of mutated versions of the current genome to generate.
        mutation: func
            The mutation function used to decide how a genome is mutated.
            
        Returns
        --------
        out: list
            A list of genomes.
        """
        from genomes import Genome
        pop = []
        
        for i in range(population):
            x = mutation(self.clone())
            pop.append(x)
            
            if not isinstance(x, Genome):
                raise Exception('Unexpected return type of mutation, got {} but expected Genome.'.format(type(x)))
                
        return pop
        
    def __str__(self):
        s = f'<Genome fitness="{self.fitness}"'
        if self.name:
            s += f' name="{self.name}"'
            
        for prop in self.props:
            val = str(self.props[prop]).replace('\n', ' ').replace('\r', '')
            s+= f' {prop}="{val}"'
            
        s+= '>'
        return s
    def __repr__(self):
        return self.__str__()
    def __getattr__(self, attr):
        if attr in self.props:
            return self.props[attr]
        else:
            raise AttributeError('\'Genome\' object has no attribute \'{}\''.format(attr))


# In[ ]:





# In[11]:


class GenomeNode:
    def __init__(self):
        self.connections = []
    
    def connect(self, targets):
        """
        Connect the current node to another node.
        
        A node cannot be connected to the same node twice, in an attempt to connect two nodes that are already connected an exception will be raised.
        
        Parameters
        -----------
        targets: GenomeNode, list[GenomeNode]
            The targets add to the current node's connections.
        
        Returns
        --------
        out: None
        """
        if utils.is_array_like(targets):
            for target in targets:
                if target not in self.connections:
                    self.connections.append(target)
                else:
                    raise Exception('{} is in the current node\'s connections.')
        else:
            if targets not in self.connections:
                self.connections.append(targets)
            else:
                raise Exception('{} is in the current node\'s connections.')
                
    def to_layer(self):
        """
        Returns a keras layer object corresponding with the current node.
        
        In cases where the current instance is of GenomeInputNode type, the return value will be an input Tensor.

        Returns
        --------
        out: keras.layers.Layer or Tensor
        """
        raise Exception('You are trying to convert a default GenomeNode into a keras layer, this is not an option, instead try using GenomeInputNode, GenomeDenseNode, etc.')
    
    @staticmethod
    def connect_nodes(nodes, targets):
        """
        Connect a list of nodes to target nodes.
        
        Parameters
        -----------
        nodes: list[GenomeNode]
            A list of nodes to connect to targets.
        targets: GenomeNode or list[GenomeNode]
            The target nodes to connect each node in the list of nodes to.
            
        Returns
        --------
        out: None
        """
        for node in nodes:
            node.connect(targets)
    
    @staticmethod
    def create_inputs(inputs):
        """
        Creates a list of GenomeInputNode.
        
        Parameters
        -----------
        inputs: int
            The number of inputs nodes to create.
        
        Returns
        --------
        out: list[GenomeInputNode]
            A list of input nodes.
        """
        return [GenomeInputNode() for i in range(inputs)]


# In[4]:


class GenomeInputNode(GenomeNode):
    def to_layer(self):
        return tf.keras.layers.Input(1)

GenomeInputNode.to_layer.__doc__ = GenomeNode.to_layer.__doc__


# In[5]:


class GenomeDenseNode(GenomeNode):
    def __init__(self, activation='relu'):
        self.activation = activation
        
    def to_layer(self):
        return tf.keras.layers.Dense(1, self.activation)

GenomeDenseNode.to_layer.__doc__ = GenomeNode.to_layer.__doc__


# In[ ]:





# In[17]:


inputs = GenomeNode.create_inputs(4)
outputs = [GenomeDenseNode('relu') for i in range(3)]

GenomeNode.connect_nodes(inputs, outputs)

