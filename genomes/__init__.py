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


# In[3]:


class GenomeModel(Genome):
    """
    Represents a genome of a keras model.
    
    A genome holds the data of a keras model without instantiating an instace, in other words, it contains the "DNA" of a model.
    
    You can apply mutations on a genome, for example a "new node" mutation can be easily applied as `genome.add_node(layer)`.
    
    An instance of the represented keras model can be created using the `.to_model()` function on a genome instance.
    """
    def __init__(self, layers=None, name=None):
        """
        Initiates a genome model.
        
        Parameters
        -----------
        layers: list of GenomeLayer, optional
            A list of GenomeLayer instances, note that the first layer must be of type "input".
        name: string
            The name of the genome, later used as the name of the keras model.
        """
        if layers and layers[0].ltype != 'input':
            raise Exception('Expected first layer to be of type "input" but instead received "{}"'.format(layers[0].ltype))
        super().__init__(name=name, layers=layers)
        
    @staticmethod
    def generate_base(inputs, hidden, outputs, activation='relu', name=None):
        """
        Generates a base genome instance based on the given parameters.
        
        Parameters
        -----------
        inputs: int
            The number of nodes in the input layer.
        hidden: int
            The number of initial hidden layers, each layer starts with a single node.
        outputs: int
            The number of nodes in the output layer.
        activation: str, optional
            The activation function for all of the initial layers.
        
        Returns
        --------
        out: GenomeModel
            A genome model.
        """
        genome = GenomeModel(name=name)
        genome.add_input_layer(inputs)
        
        for i in range(hidden):
            genome.add_hidden_layer(1, activation)
            
        genome.add_output_layer(outputs, activation)
        
        genome.connect_layers() # Find appropriate name
        genome.fill_biases() # Find appropriate name
        
        return genome
    
    def __str__(self):
        s = f'<GenomeModel fitness={self.fitness}'
        if self.layers and len(self.layers):
            s += f' inputs="{self.layers[0].units}" outputs="{self.layers[-1].units}"'
        if self.name:
            s+= f' name="{self.name}"'
        return s + '>'
    def __repr__(self):
        return self.__str__()


# In[24]:


class GenomeLayer:
    """
    Represents a layer in a genome model.
    """
    def __init__(self, units, activation=None, weights=None, bias=None, ltype='dense', weights_initializer='glorot_uniform', bias_initializer='glorot_uniform', name=None):
        """
        Initiates a genome layer.
        
        Parameters
        -----------
        units: int
            The number of units in the layer.
        activation: string or function, optional
            The activation function of the layer, this is optional for layers of type `input`.
        weights: array or list, optional
            The connections to the outbounding layer.
        bias: array or list, optional
            The biases of the layer.
        ltype: str, optional
            The type of a keras layer, currently only input/dense are supported.
        weights_initializer: str or function, optional
            Function used to initialize or fill missing connections (weights) with the outbounding layer.
        bias_initializer: str or function, optional
            Function used to initialize or fill missing biases in the current layer.
        name: str, optional
            The name of the genome layer, later used as the name of the keras layer.
        """
        self.ltype = ltype.lower()
        if self.ltype not in ['input', 'dense']:
            raise Exception('Unsupported type of genome layer received "{}", expected types are input/dense.'.format(self.ltype))
        self.weights_initializer = tf.keras.initializers.get(weights_initializer) if type(weights_initializer) is str else weights_initializer
        self.bias_initializer = tf.keras.initializers.get(bias_initializer) if type(bias_initializer) is str else bias_initializer
        self.units = units
        self.activation = activation
        self.weights = np.array(weights) if weights else None
        self.bias = np.array(bias) if bias else None
        self.name = name
        
    @staticmethod
    def create_input(units, name=None):
        """
        Creates an input genome layer.

        Parameters
        -----------
        units: int
            The number input units.

        Returns
        --------
        out: GenomeLayer
            An input genome layer.
        """
        return GenomeLayer(units, ltype='input', name=name)
        
    def connect_layer(self, outbound_layer): # Find appropriate name & use
        """
        Naively connect the weights of the current layer to the given outbounding layer.
        The missing weights are filled using the `weights_initializer` function while the unused weights are removed.
        
        Parameters
        -----------
        outbound_layer: GenomeLayer
            The outbounding layer.
        
        Returns
        --------
        out: None
        """
        import utils
        new_weights_shape = self.units, outbound_layer.units
        weights = self.weights if self.weights else [[]]
        new_weights = self.weights_initializer(new_weights_shape)
        self.weights = utils.fill_array(new_weights, weights)
    
    def __str__(self):
        s = f'<GenomeLayer type="{self.ltype}" units="{self.units}"'
        if self.ltype != 'input':
            s += f' activation="{self.activation}"'
        if self.name:
            s+= f' name="{self.name}"'
        return s + '>'
    def __repr__(self):
        return self.__str__()


# In[25]:


inputs = GenomeLayer.create_input(5)
outputs = GenomeLayer()

