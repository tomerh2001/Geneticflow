#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os, sys
sys.path.append(os.path.realpath('..'))

import copy

import utils


# In[19]:


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





# In[43]:


class GenomeNode:
    """
    This is the base class for all nodes in a partially connected neural network (PCNN).
    
    Nodes can be connected to any other node in the PCNN, each connection is stored along with the weights and biases.
    """
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
    
    def remove_all_connections(self):
        """
        Removes all connections of the current node.
        """
        del self.connections[:]
    
    def remove_connection(self, node):
        """
        Removes the given node from the current node's connections, including the associated weights and biases.
        
        Parameters
        -----------
        node: GenomeNode
            The node to remove from the connections of the current node.
            
        Returns
        --------
        out: None
        """
        if not issubclass(type(node), GenomeNode):
            raise Exception('Expected node to be of type GenomeNode, instead received {}'.format(type(node)))
        self.connections.remove(node)
                
    def to_layer(self):
        """
        Returns a keras layer object corresponding with the current node.
        
        In cases where the current instance is of GenomeInputNode type, the return value will be an input Tensor.

        Returns
        --------
        out: keras.layers.Layer or Tensor
        """
        raise Exception('You are trying to convert a default GenomeNode into a keras layer, this is not an option, instead try using GenomeInputNode, GenomeDenseNode, etc.')


# In[21]:


class GenomeInputNode(GenomeNode):
    """
    An input node in a partially connected neural netowrk (PCNN).
    
    This is used to represent a keras input layer with a single unit.
    """
    def to_layer(self):
        return tf.keras.layers.Input(1)

GenomeInputNode.to_layer.__doc__ = GenomeNode.to_layer.__doc__


# In[22]:


class GenomeDenseNode(GenomeNode):
    """
    A dense node in a partially connected neural netowrk (PCNN).
    
    Unlike a Dense layer in keras, this class represents a single node (dense layer with a single unit) that is'nt densly connected to all the following nodes, but rather partially connected, meaning that the node can be connected to any other node in the entire network.
    
    This is the base node used in the PCNN.
    """
    def __init__(self, activation='relu'):
        self.activation = activation
        
    def to_layer(self):
        return tf.keras.layers.Dense(1, self.activation)

GenomeDenseNode.to_layer.__doc__ = GenomeNode.to_layer.__doc__


# In[23]:


class GenomeLayer:
    """
    This is the base class for all layers in a partially connected neural network (PCNN).
    
    Each layer contains nodes (instances of `GenomeNode`).
    """
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


# In[28]:


class GenomeInputLayer(GenomeLayer):
    """
    An input layer in a partially connected neural netowrk (PCNN).
    
    A GenomeInputLayer contains a list of GenomeInputNode instances.
    
    This is usually the first layer in a PCNN.
    """
    def __init__(self, inputs=1):
        super().__init__(inputs)
        
    def create_node(self):
        return GenomeInputNode()


# In[29]:





# In[37]:


class GenomePCNN(Genome):
    """
    Partially Connected Neural Netowrk (PCNN).
    
    This class is used as a genome of a NN model for the NEAT algorithm.
    
    To convert an instance into a keras model use the `.to_model` function.
    """
    def __init__(self, layers, name=None):
        for layer in layers:
            if not issubclass(type(layer), GenomeLayer):
                raise Exception('Layers of PCNN must be of type GenomeLayer. (received {})'.format(type(layer)))
        
        super().__init__(name=name, layers=layers)

    def __str__(self):
        return super().__str__(False)


# In[ ]:


model = GenomePCNN.create_base(4, 4)
model.layers[0].nodes[0].connections


# In[ ]:




