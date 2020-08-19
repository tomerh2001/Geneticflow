#!/usr/bin/env python
# coding: utf-8

# In[34]:


import tensorflow as tf
import numpy as np


# In[113]:


class GenomeLayer:
    """
    Represents a layer in a genome model.
    
    An instance WITHOUT WEIGHTS of the represented keras layer can be created using the `.to_layer()` function on a genome instance.
    """
    def __init__(self, units, activation, weights=None, bias=None, ltype='dense', weights_filler='glorot_uniform', bias_filler='glorot_uniform', name=None):
        """
        Initiates a genome layer.
        
        Parameters
        -----------
        units: int
            The number of units in the layer.
        activation: string or function
            The activation function of the layer.
        weights: array or list, optional
            The connections to the outbounding layer.
        bias: array or list, optional
            The biases of the layer.
        ltype: str, optional
            The type of a keras layer, currently only input/dense are supported.
        weights_filler: str or function, optional
            Function used to fill missing connections with the outbounding layer. 
        bias_filler: str or function, optional
            Function used to fill missing biases in the current layer.
        name: str, optional
            The name of the genome layer, later used as the name of the keras layer.
        """
        self.ltype = ltype.lower()
        if self.ltype not in ['input', 'dense']:
            raise Exception('Unsupported type of genome layer received "{}", expected types are input/dense.'.format(self.ltype))
        self.weights_filler = tf.keras.initializers.get(weights_filler) if type(weights_filler) is str else weights_filler
        self.bias_filler = tf.keras.initializers.get(bias_filler) if type(bias_filler) is str else bias_filler
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
        return GenomeLayer(units, None, None, None, ltype='input', name=name)
    
    def to_layer(self, inputs=None):
        """
        Converts the current genome layer to a keras layer WITHOUT THE WEIGHTS and connects it to a preceding layer (for types other than "input").
        NOTE: The keras layer is created WITHOUT THE WEIGHTS OF THE GENOME, meaning that it is created with RANDOM WEIGHTS, the weights of the genome are attached to the keras layer
        only when creating a keras model with `GenomeModel.to_model`. 
        
        Parameters
        -----------
        inputs: keras.layers.Layer, optional
            The preceding keras layer, this parameters is not needed for layers of type "input".
            
        Returns
        --------
        out: keras.layers.Layer
            A keras layer which is connected to the provided preceding layer.
        """
        if self.ltype == 'input':
            return tf.keras.layers.InputLayer(self.units, name=self.name)
        elif self.ltype == 'dense':
            layer = tf.keras.layers.Dense(self.units, activation=self.activation, name=self.name)
            layer(inputs.output)
            return layer
        
    def connect_layer(self, outbound_layer):
        import utils
        """
        Naively connect the weights of the current layer to the given outbounding layer.
        The missing weights are filled using the `weights_filler` function while the unused weights are removed.
        
        Parameters
        -----------
        outbound_layer: GenomeLayer
            The outbounding layer.
        
        Returns
        --------
        out: None
        """
        new_weights_shape = self.units, outbound_layer.units
        weights = self.weights if self.weights else [[]]
        new_weights = self.weights_filler(new_weights_shape)
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


# In[112]:


class GenomeModel:
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
        self.layers = layers
        self.name = name
        self.fitness = 0
        
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
        
        genome.connect_layers()
        genome.fill_biases()
        
        return genome
    
    def to_model(self):
        """
        Converts the current genome model to a keras model.
        
        Returns
        --------
        out: keras.layers.Model
            A keras model.
        """        
        inputs_genome = self.layers[0]
        inputs_layer = prev_layer = inputs_genome.to_layer()
        
        weights = []
        weights.extend([inputs_genome.weights, inputs_genome.bias])
        
        for layer_genome in self.layers[1:-1]:
            weights.extend([layer_genome.weights, layer_genome.bias])
            layer_genome.to_layer(prev_layer)
            prev_layer = layer_genome
        
        outputs_genome = self.layers[-1]
        outputs_layer = outputs_genome.to_layer(prev_layer)
        
        model = tf.keras.Model(inputs=inputs_layer.input, outputs=outputs_layer.output, name=self.name)
        model.set_weights(weights)
        return model
    
    def __str__(self):
        s = f'<GenomeModel'
        if self.layers and len(self.layers):
            s += f' inputs="{self.layers[0].units}" outputs="{self.layers[-1].units}"'
        if self.name:
            s+= f' name="{self.name}"'
        return s + '>'
    def __repr__(self):
        return self.__str__()


# In[ ]:





# In[ ]:


# Todo
# Decide how to handle the weights of the layers.
# Decide how the fill & clear connections.
# Decide when to apply fill & clear.
# Add mutations module with `mutate_genome`

