#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
sys.path.append(os.path.realpath('..'))

import copy
import inspect
from tensorflow import keras
from tensorflow.keras import initializers

import utils


# In[2]:


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





# In[3]:


class NodeConnection:
    def __init__(self, input_node, output_node, w, b=None):
        self.input_node = input_node
        self.output_node = output_node
        self.w = w
        self.b = b


# In[22]:


class InputNode:
    def __init__(self, w_initializer='glorot_uniform', name=None):
        self.connections = { 'inputs': [], 'outputs': [] }
        
        if type(w_initializer) is str:
            self.w_initializer = initializers.get(w_initializer)
        elif inspect.isfunction(w_initializer):
            self.w_initializer = w_initializer
        else:
            raise Exception('Expected w_initializer to be either a function or a function name, instead got {}.'.format(type(w_initializer)))

        self.name = name
        self.layer = self.to_layer()
            
    def connect(self, node):
        if not isinstance(node, DenseNode):
            raise Exception('Node must be of type DenseNode, instead got {}.'.format(type(node)))
        elif node == self:
            raise Exception('An attempt to connect a node to itself has been made.')
        elif node in [c.output_node for c in self.connections['outputs']]:
            raise Exception('An attempt to connect two nodes that are already connected has been made.')
        else:
            w = self.w_initializer([1])
            c = NodeConnection(self, node, w)
            self.connections['outputs'].append(c)
            node.connections['inputs'].append(c)
            
    def to_layer(self):
        return keras.layers.InputLayer((1, ), name=self.name)


# In[24]:


class DenseNode(InputNode):
    def __init__(self, activation='relu', w_initializer='glorot_uniform', b_initializer='zeros', name=None):
        if type(b_initializer) is str:
            self.b_initializer = initializers.get(b_initializer)
        elif inspect.isfunction(b_initializer):
            self.b_initializer = b_initializer
        else:
            raise Exception('Expected b_initializer to be either a function or a function name, instead got {}.'.format(type(b_initializer)))
            
        self.activation = activation
        super().__init__(w_initializer, name=name)

    def connect(self, node):
        if not isinstance(node, DenseNode):
            raise Exception('Node must be of type DenseNode, instead got {}.'.format(type(node)))
        elif node == self:
            raise Exception('An attempt to connect a node to itself has been made.')
        elif node in [c.output_node for c in self.connections['outputs']]:
            raise Exception('An attempt to connect two nodes that are already connected has been made.')
        else:
            w = self.w_initializer([1])
            b = self.b_initializer([1])
            c = NodeConnection(self, node, w, b)
            self.connections['outputs'].append(c)
            node.connections['inputs'].append(c)
            
    def to_layer(self):
        return keras.layers.Dense(1, self.activation, name=self.name)


# In[25]:


class GenomePCNN(Genome):
    def __init__(self, inputs, outputs, name=None):
        for node in inputs:
            if type(node) is not InputNode:
                raise Exception('All inputs must be of type InputNode, instead got {}.'.format(type(node)))
                
        for node in outputs:
            if type(node) is not DenseNode:
                raise Exception('All outputs must be of type DenseNode, instead got {}.'.format(type(node)))
        
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        
    @staticmethod
    def create_blank(inputs, outputs, name=None):
        input_nodes = [InputNode() for i in range(inputs)]
        output_nodes = [DenseNode() for i in range(outputs)]
        
        for input_node in input_nodes:
            for output_node in output_nodes:
                if utils.chance(.5):
                    input_node.connect(output_node)
                    
        return GenomePCNN(input_nodes, output_nodes, name=name)
    
    def to_model(self):
        def traverse_and_connect(connections, tensor):
            for connection in connections:
                x = connection.output_node.layer(tensor)
                traverse_and_connect(connection.output_node.connections, x)

        for node in self.inputs:
            traverse_and_connect(node.connections, node.layer.input)

        return keras.Model(inputs=[x.layer.input for x in self.inputs], outputs=[x.layer.output for x in self.outputs])
    
    def __str__(self, with_props=True):
        s = f'<GenomePCNN fitness="{self.fitness}"'
        if self.name:
            s += f' name="{self.name}"'
        s+= f' inputs="{len(self.inputs)}" outputs="{len(self.outputs)}">'
        return s


# In[58]:


l1 = [InputNode() for i in range(2)]
l2 = [DenseNode() for i in range(2)]
l3 = [DenseNode() for i in range(1)]

l1[0].connect(l2[0])
l1[0].connect(l2[1])
l1[1].connect(l2[0])
l1[1].connect(l2[1])
l2[0].connect(l3[0])
l2[1].connect(l3[0])

connected_nodes = []
def traverse_and_connect(node):
    for connection in node.connections['outputs']:
        if connection.output_node not in connected_nodes:
            connection_node_inputs = connection.output_node.connections['inputs']
            
            if len(connection_node_inputs) > 1:
                merge_layer = keras.layers.Concatenate()
                x = merge_layer([input_connection.input_node.layer for input_connection in connection_node_inputs])
                connection.output_node.layer(x)
            else:
                connection.output_node.layer(node.layer.output)
                connected_nodes.append(connection.output_node)
            
            traverse_and_connect(connection.output_node)
            
for input_node in l1:
    traverse_and_connect(input_node)


# In[ ]:





# In[ ]:





# In[ ]:




