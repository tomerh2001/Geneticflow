#!/usr/bin/env python
# coding: utf-8

# In[14]:


import tensorflow as tf
import numpy as np
import copy


# In[12]:


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
        return [mutation(self.clone()) for i in range(population)]
        
    def __str__(self):
        s = f'<Genome fitness="{self.fitness}"'
        if self.name:
            s += f' name="{self.name}"'
        s+= '>'
        return s
    def __repr__(self):
        return self.__str__()
    def __getattr__(self, attr):
        if attr in self.props:
            return self.props[attr]
        else:
            raise AttributeError('\'Genome\' object has no attribute \'{}\''.format(attr))


# In[62]:


class GenomeNode:
    def __init__(self):
        self.outputs = []
    
    def connect_layer(self, layer):
        self.outputs.append(layer)
    
    def __str__(self):
        return f'<{type(self).__name__}>'
    def __repr__(self):
        return self.__str__()


# In[64]:


class GenomeInputNode(GenomeNode):
    def __init__(self):
        super().__init__()
        
    def to_layer(self, shape=1):
        return tf.keras.layers.InputLayer(shape)


# In[65]:


class GenomeDenseNode(GenomeNode):
    def __init__(self, activation='relu'):
        super().__init__()
        self.activation = activation
        
    def to_layer(self):
        return tf.keras.layers.Dense(1, self.activation)


# In[69]:


# input1 = GenomeInputNode()
# output1 = GenomeDenseNode()

# input1.connect_layer(output1)

# input1.outputs


# In[ ]:


# # Option 1
# # Connect all nodes to target nodes
# # Target can be a layer or a single node.
# inputs.connect_layer(hidden)

# # Option 2
# # Connect node i to target nodes
# # Target can be a layer or a single node.
# inputs.nodes[i].connect_node(hidden)

