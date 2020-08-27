#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os, sys
sys.path.append(os.path.realpath('..'))

from tensorflow import keras
import numpy as np


# In[5]:


def chance(rate, size=None):
    """
    Returns true based on the propabilty (rate) provided.
    
    Parameters
    -----------
    rate: int
        A number between 0-1 that represents a probability.
    size: int or tuple, optional
        The shape of the output.
    
    Returns
    --------
    out: bool
    """
    return np.random.random_sample(size) < rate


# In[7]:


def fill_array(a, b):
    """
    Fill an array with another array while maintaining original values where they have not been overwritten.
    
    Parameters
    -----------
    a: array_like
        An array.
    b: array_like
        An array.
    
    Returns
    --------
    out: ndarray
        A copy of the first array where values from the second array are overwritten in the shared dimensions between the two arrays.
        
    Examples
    ---------
    >>> a = np.zeros((3, 3))
    >>> b = np.ones((2, 2))
    >>> fill_array(a, b)
    array([[1., 1., 0.],
           [1., 1., 0.],
           [0., 0., 0.]])
    """
    a, b = np.array(a), np.array(b)
    x1, x2 = np.min((a.shape, b.shape), 0)
    c = a.copy()
    c[:x1, :x2] = b[:x1, :x2]
    return c


# In[8]:


def evenly_combine(objects, select=None):
    """
    Evenly combine multiple objects into a single object.
    
    Parameters
    -----------
    values: list of objects
        A list of objects to combine, the objects can be of any type as long as the selected object is either an iterable or a dict.
    select: func
        A selection function which is used to select the value (iterable or dict) from an object.
    
    Returns
    --------
    out: ndarray or dict
        An object containing values from all other objects.
        
    Examples
    ---------
    >>> a = [1, 1, 1]
    >>> b = [2, 2, 2]
    >>> c = [3, 3, 3]
    >>> evenly_combine((a, b, c))
    array([1, 2, 3])
    
    >>> a = {'a': 1, 'b': 1, 'c': 1}
    >>> b = {'a': 2, 'b': 2, 'c': 2}
    >>> c = {'a': 3, 'b': 3, 'c': 3}
    >>> evenly_combine((a, b, c))
    {'a': 1, 'b': 2, 'c': 3}
    
    >>> a = [1, 1, 1]
    >>> b = [2, 2, 2]
    >>> evenly_combine((a, b))
    array([1, 1, 2])
    """
    if not select:
        select = lambda x: x

    size = len(select(objects[0]))
    vals_per_object = np.ceil(size / len(objects)).astype(int)
    
    first_object = select(objects[0])
    
    if type(first_object) is dict:
        new_objects = {}
        for i, obj in enumerate(first_object):
            new_objects[obj] = select(objects[i//vals_per_object])[obj]
    else:
        new_objects = []
        for i, obj in enumerate(objects):
            start_index = vals_per_object*i
            end_index = start_index + vals_per_object
            new_objects.extend(select(obj)[start_index:end_index])
        new_objects = np.array(new_objects)
        
    return new_objects


# In[ ]:




