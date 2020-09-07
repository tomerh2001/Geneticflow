#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os, sys
sys.path.append(os.path.realpath('..'))

import tensorflow as tf
import numpy as np
import copy

import utils


# In[8]:


class Genome:
    """
    Represents a genome. Used to hold the "DNA" of an object.
    This is the base class for all other genomes.
    
    Examples
    ---------
    >>> # Create a genome with the properties {a = 1, b = 2, c = 3}
    >>> gen = Genome(a=1, b=2, c=3)
    <Genome fitness="0" a="1" b="2" c="3">
    
    >>> # Access the genome property `a`
    >>> gen.a
    >>> 1
    
    >>> # Access and change the genome fitness
    >>> gen.fitness = 50
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
        
    def __str__(self, with_props=True):
        s = f'<Genome fitness="{self.fitness}"'
        if self.name:
            s += f' name="{self.name}"'
        
        if with_props:
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


# In[15]:


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
    
    def remove_connections(self):
        """
        Removes all connections of the current node.
        """
        del self.connections[:]
                
    def to_layer(self):
        """
        Returns a keras layer object corresponding with the current node.
        
        In cases where the current instance is of GenomeInputNode type, the return value will be an input Tensor.

        Returns
        --------
        out: keras.layers.Layer or Tensor
        """
        raise Exception('You are trying to convert a default GenomeNode into a keras layer, this is not an option, instead try using GenomeInputNode, GenomeDenseNode, etc.')


# In[16]:


class GenomeInputNode(GenomeNode):
    def to_layer(self):
        return tf.keras.layers.Input(1)

GenomeInputNode.to_layer.__doc__ = GenomeNode.to_layer.__doc__


# In[17]:


class GenomeDenseNode(GenomeNode):
    def __init__(self, activation='relu'):
        self.activation = activation
        
    def to_layer(self):
        return tf.keras.layers.Dense(1, self.activation)

GenomeDenseNode.to_layer.__doc__ = GenomeNode.to_layer.__doc__


# In[18]:


class GenomeLayer:
    def __init__(self, nodes=1):
        self.nodes = [self.create_node() for i in range(nodes)]
        
    def connect(self, targets):
        """
        Connects all nodes of the current layer to all nodes of another layer.
        
        Parameters
        -----------
        targets: GenomeLayer
            A layer to connect all the nodes of the current layer to.
            
        Returns
        --------
        out: None
        """
        for node in self.nodes:
            for target_node in targets.nodes:
                node.connect(target_node)
    
    def create_node(self):
        """
        Creates and returns a new node based on the current layer (i.e. GenomeDenseLayer will return a GenomeDenseNode while GenomeInputLayer will return a GenomeInputNode).
        
        Returns
        --------
        out: GenomeNode
            A node based on the type of the layer.
        """
        raise Exception('You are trying to use the GenomeLayer class directly, this is not an option, please check out GenomeInputLayer, GenomeDenseLayer, etc instead.')


# In[7]:


class GenomeInputLayer(GenomeLayer):
    def __init__(self, inputs=1):
        super().__init__(inputs)
        
    def create_node(self):
        return GenomeInputNode()


# In[8]:


class GenomeDenseLayer(GenomeLayer):
    def __init__(self, nodes=1, activation='relu'):
        self.activation = activation
        super().__init__(nodes)
        
    def create_node(self):
        return GenomeDenseNode(self.activation)


# In[19]:


class GenomePCNN(Genome):
    def __init__(self, inputs, outputs, name=None):
        super().__init__(name=name, layers=[])
        input_layer, output_layer = GenomeInputLayer(inputs), GenomeDenseLayer(outputs)
        self.layers.extend([input_layer, output_layer])
        
    def __str__(self):
        return super().__str__(False)


# In[23]:


model = GenomePCNN(inputs=4, outputs=4)

# model.add_hidden_layer(1, 'relu')

model.layers

