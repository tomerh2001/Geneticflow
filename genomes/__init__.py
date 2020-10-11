#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os, sys
sys.path.append(os.path.realpath('..'))

import copy
import inspect
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras.models import Model

import utils


# In[12]:


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


# In[ ]:





# In[13]:


class InputNode(keras.layers.InputLayer):
    def __init__(self):
        super().__init__(1)
        
    @property
    def output_tensor(self):
        return self.output[0]


# In[14]:


class Node(keras.layers.Layer):
    def __init__(self, activation='relu', kernel_init="glorot_uniform", bias_init="zeros", name=None):
        super().__init__(trainable=False, name=name, dynamic=True)
        self.activation = keras.activations.get(activation)
        self.kernel_init = keras.initializers.get(kernel_init)
        self.bias_init = keras.initializers.get(bias_init)
        self.bias = tf.Variable(self.bias_init(1)[0], trainable=False, name='bias')
        self.inputs = None
        self.kernel = None
    
    def connect(self, node):
        """
        Connects the current node to the given node while performing basic connection tasks such as:
        1. Adding the given node to the `inbound_nodes` of the current node and adding the current node to the `outbound_nodes` of the given node.
        2. Adding the output tensor of the given node to the inputs of the current node.
        3. Adding a new weight to the kernel of the current node.
        
        Parameters
        -----------
        node: Node or InputNode
            A node to add to the inputs of the current node.
        
        Returns
        --------
        out: None
        """
        if not (issubclass(type(node), (Node, InputNode))):
            raise Exception('An attempt to connect {} with a {} has been made, expected Node or InputNode.'.format(self, type(node)))
        
        self.inbound_nodes.append(node)
        node.outbound_nodes.append(self)
        
        if not tf.is_tensor(self.inputs):
            self.inputs = node.output_tensor
        else:
            if type(node) is InputNode:
                self.inputs = tf.concat([self.inputs, node.output_tensor], 0)
            else:
                self.inputs = tf.stack([self.inputs, node.output_tensor], 0)
        
        w = self.kernel_init([1])
        if not tf.is_tensor(self.kernel):
            self.kernel = tf.Variable(w, trainable=False, name='kernel')
        else:
            self.kernel = tf.concat([self.kernel, w], 0)
        
    def call(self, nodes):
        if utils.is_array_like(nodes):
            for node in nodes:
                self.connect(node)
        else:
            self.connect(nodes)
        
        x = self.activation(tf.tensordot(self.inputs, self.kernel, 1) + self.bias)
        self.output_tensor = tf.identity(x, name=f'{self.name}_output')
        return self.output_tensor


# In[24]:


class GenomePCNN(Genome, Model):
    def __init__(self, input_nodes, output_nodes, name='PCNN'):
        super(Model, self).__init__(inputs=[x.input for x in input_nodes], outputs=[x.output_tensor for x in output_tensor], name=name)
        


# In[ ]:





# In[ ]:





# In[23]:


# Test 1
# ------
import numpy as np
relu = keras.activations.relu

# Nodes
i1, i2, i3 = InputNode(), InputNode(), InputNode()
n1, n2 = Node(), Node()
o1 = Node()

n1(i1)
n2([i1, i2, i3])
o1([n1, n2, i3])

# Model
model = keras.Model(inputs=[i1.input, i2.input, i3.input], outputs=[o1.output_tensor])
 
# Weights
w1 = n1.kernel.numpy()[0]
w2, w3, w4 = n2.kernel.numpy()
w5, w6, w7 = o1.kernel.numpy()

# Input Data
a, b, c = np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))

# Self-check of formula
x1 = relu(a * w1)
x2 = relu(a * w2 + b * w3 + c * w4)
x3 = relu(x1 * w5 + x2 * w6 + c * w7)

# Test prediction
x_pred = model.predict([a, b, c])
np.round(x_pred, 5) == np.round(x3.numpy()[0, 0], 5)


# In[ ]:




