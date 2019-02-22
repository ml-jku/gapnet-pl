# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017
Different classes and utility functions for stack-able network layers

See architectures/sample_architectures.py for some usage examples

"""

import numbers

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


# ------------------------------------------------------------------------------------------------------------------
#  Imports
# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------
#  Functions
# ------------------------------------------------------------------------------------------------------------------
def tof(i, shape):
    """Check whether i is tensor or initialization function; return tensor or initialized tensor;
    
    Parameters
    -------
    i : tensor or function
        Tensor or function to initialize tensor
    shape : list or tuple
        Shape of tensor to initialize
    
    Returns
    -------
    : tensor
        Tensor or initialized tensor
    """
    if callable(i):
        return i(shape)
    else:
        return i


def tofov(i, shape=None, var_params=None):
    """Check whether i is tensor or initialization function or tf.Variable; return tf.Variable;
    
    Parameters
    -------
    i : tensor or function or tf.Variable
        Tensor or function to initialize tensor
    shape : list or tuple or None
        Shape of tensor to initialize
    var_params : dict or None
        Dictionary with additional parameters for tf.Variable, e.g. dict(trainable=True); Defaults to empty dict;
    
    Returns
    -------
    : tf.Variable
        Tensor or initialized tensor or tf.Variable as tf.Variable
    """
    
    if isinstance(i, tf.Variable):
        # i is already a tf.Variable -> nothing to do
        return i
    else:
        # i is either a tensor or initializer -> turn it into a tensor with tof()
        i = tof(i, shape)
        # and then turn it into a tf.Variable
        if var_params is None:
            var_params = dict()
        return tf.Variable(i, **var_params)


def dot_product(tensor_nd, tensor_2d):
    """Broadcastable version of tensorflow dot product between tensor_nd ad tensor_2d
    
    Parameters
    -------
    tensor_nd : tensor
        Tensor with 1, 2 or more dimensions; Dot product will be performed on last dimension and broadcasted over other
        dimensions
    tensor_2d : tensor
        Tensor with 1 or 2 dimensions;
    
    Returns
    -------
    : tensor
        Tensor for dot product result
    """
    # Get shape and replace unknown shapes (None) with -1
    shape_nd = tensor_nd.get_shape().as_list()
    shape_nd = [s if isinstance(s, int) else -1 for s in shape_nd]
    shape_2d = tensor_2d.get_shape().as_list()
    if len(shape_2d) > 2:
        raise ValueError("tensor_2d must be a 1D or 2D tensor")
    
    if len(shape_2d) == 1:
        tensor_2d = tf.expand_dims(tensor_2d, 0)
    if len(shape_nd) == 1:
        shape_nd = tf.expand_dims(shape_nd, 0)
    
    if len(shape_nd) > 2:
        # collapse axes except for ones to multiply and perform matmul
        dot_prod = tf.matmul(tf.reshape(tensor_nd, [-1, shape_nd[-1]]), tensor_2d)
        # reshape to correct dimensions
        dot_prod = tf.reshape(dot_prod, shape_nd[:-1] + shape_2d[-1:])
    elif len(shape_nd) == 2:
        dot_prod = tf.matmul(tensor_nd, tensor_2d)
    else:
        dot_prod = tf.matmul(tf.expand_dims(tensor_nd, 0), tensor_2d)
    
    return dot_prod


def conv2d(x, W, strides=(1, 1, 1, 1), padding='SAME', dilation_rate=(1, 1), name='conv2d'):
    """Broadcastable version of tensorflow 2D convolution with weight mask, striding, zero-padding, and dilation
    
    For dilation the tf.nn.convolution function is used. Otherwise the computation will default to the (cudnn-
    supported) tf.nn.conv2d function.
    
    Parameters
    -------
    x : tensor
        Input tensor to be convoluted with weight mask; Shape can be [samples, x_dim, y_dim, features] or
        [samples, timesteps, x_dim, y_dim, features]; Convolution is performed over last 3 dimensions;
    W : tensor
        Kernel to perform convolution with; Shape: [x_dim, y_dim, input_features, output_features]
    padding : str or tuple of int
        Padding method for image edges (see tensorflow convolution for further details); If specified as
        tuple or list of integer tf.pad is used to symmetrically zero-pad the x and y dimensions of the input.
        Furthermore supports TensorFlow paddings "VALID" and "SAME" in addition to "ZEROPAD" which symmetrically
        zero-pads the input so output-size = input-size / stride (taking into account strides and dilation;
        comparable to Caffe and Theano).
    dilation_rate : tuple of int
        Defaults to (1, 1) (i.e. normal 2D convolution). Use list of integers to specify multiple dilation rates;
        only for spatial dimensions -> len(dilation_rate) must be 2;
    
    Returns
    -------
    : tensor
        Tensor for convolution result
    """
    x_shape = x.get_shape().as_list()
    x_shape = [s if isinstance(s, int) else -1 for s in x_shape]
    W_shape = W.get_shape().as_list()
    padding_x = None
    padding_y = None
    
    if padding == "ZEROPAD":
        if len(x_shape) == 5:
            s = strides[1:3]
            i = (int(x_shape[2] / s[0]), int(x_shape[3] / s[1]))
        elif len(x_shape) == 4:
            s = strides[1:3]
            i = (int(x_shape[1] / s[0]), int(x_shape[2] / s[1]))
        else:
            raise ValueError("invalid input shape")
        # --
        kernel_x = W_shape[0]
        kernel_y = W_shape[1]
        padding_x = int(np.ceil((i[0] - s[0] - i[0] + kernel_x + (kernel_x - 1) * (dilation_rate[0] - 1)) / (s[0] * 2)))
        padding_y = int(np.ceil((i[1] - s[1] - i[1] + kernel_y + (kernel_y - 1) * (dilation_rate[1] - 1)) / (s[1] * 2)))
    elif (isinstance(padding, list) or isinstance(padding, tuple)) and len(padding) == 2:
        padding_x = padding[0]
        padding_y = padding[1]
    
    if padding_x is not None and padding_y is not None:
        if len(x_shape) == 5:
            pad = [[0, 0], [0, 0], [padding_x, padding_x], [padding_y, padding_y], [0, 0]]
        elif len(x_shape) == 4:
            pad = [[0, 0], [padding_x, padding_x], [padding_y, padding_y], [0, 0]]
        
        # pad input with zeros
        x = tf.pad(x, pad, "CONSTANT")
        # set padding method for convolutions to valid to not add additional padding
        padding = "VALID"
    elif padding not in ("SAME", "VALID"):
        raise ValueError("unsupported padding type")
    
    if dilation_rate == (1, 1):
        def conv_fct(inp):
            return tf.nn.conv2d(input=inp, filter=W, padding=padding, strides=strides, name=name)
    else:
        if (strides[0] != 1) or (strides[-1] != 1):
            raise AttributeError("Striding in combination with dilation is only possible along the spatial dimensions,"
                                 "i.e. strides[0] and strides[-1] have to be 1.")
        
        def conv_fct(inp):
            return tf.nn.convolution(input=inp, filter=W, dilation_rate=dilation_rate,
                                     padding=padding, strides=strides[1:3], name=name)
    
    # Flatten matrix in first dimensions if necessary (join samples and sequence positions)
    with tf.variable_scope(name):
        if len(x_shape) > 4:
            x_shape = [s if isinstance(s, int) else -1 for s in x.get_shape().as_list()]
            if x_shape[0] == -1 or x_shape[1] == -1:
                x_flat = tf.reshape(x, [-1] + x_shape[2:])
            else:
                x_flat = tf.reshape(x, [x_shape[0] * x_shape[1]] + x_shape[2:])
            conv = conv_fct(x_flat)
            conv = tf.reshape(conv, x_shape[:2] + conv.get_shape().as_list()[1:])
        else:
            conv = conv_fct(x)
    return conv


def avgpool2D(x, ksize, strides, padding, data_format):
    """Broadcastable version of tensorflow max_pool
    
    Parameters
    -------
    x : tensor
        Input tensor to be convoluted with weight mask; Shape can be [samples, x_dim, y_dim, features] or
        [samples, timesteps, x_dim, y_dim, features]; Convolution is performed over last 3 dimensions;
    
    Returns
    -------
    : tensor
        Tensor for avgpooling result
    """
    x_shape = x.get_shape().as_list()
    x_shape = [s if isinstance(s, int) else -1 for s in x_shape]
    
    # Flatten matrix in first dimensions if necessary (join samples and sequence positions)
    if len(x_shape) > 4:
        if x_shape[0] == -1:
            x_flat = tf.reshape(x, [-1] + x_shape[2:])
        else:
            x_flat = tf.reshape(x, [x_shape[0] * x_shape[1]] + x_shape[2:])
        avgpool = tf.nn.avg_pool(x_flat, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
        avgpool = tf.reshape(avgpool, x_shape[:2] + avgpool.get_shape().as_list()[1:])
    else:
        avgpool = tf.nn.avg_pool(x, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
    return avgpool


def maxpool2D(x, ksize, strides, padding, data_format):
    """Broadcastable version of tensorflow max_pool
    
    Parameters
    -------
    x : tensor
        Input tensor to be convoluted with weight mask; Shape can be [samples, x_dim, y_dim, features] or
        [samples, timesteps, x_dim, y_dim, features]; Convolution is performed over last 3 dimensions;
    
    Returns
    -------
    : tensor
        Tensor for maxpooling result
    """
    x_shape = x.get_shape().as_list()
    x_shape = [s if isinstance(s, int) else -1 for s in x_shape]
    
    # Flatten matrix in first dimensions if necessary (join samples and sequence positions)
    if len(x_shape) > 4:
        if x_shape[0] == -1:
            x_flat = tf.reshape(x, [-1] + x_shape[2:])
        else:
            x_flat = tf.reshape(x, [x_shape[0] * x_shape[1]] + x_shape[2:])
        maxpool = tf.nn.max_pool(x_flat, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
        maxpool = tf.reshape(maxpool, x_shape[:2] + maxpool.get_shape().as_list()[1:])
    else:
        maxpool = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
    return maxpool


def get_input(incoming):
    """Get input from Layer class or tensor
    
    Check if input is available via get_output() function or turn tensor into lambda expressions instead; Also
    try to fetch shape of incoming via get_output_shape();
    
    Returns
    -------
    : tensor
        Tensor with input
    : list
        Shape of input tensor
    """
    try:
        return incoming.get_output, incoming.get_output_shape()
    except AttributeError:
        return lambda **kwargs: incoming, [d if isinstance(d, int) else -1 for d in incoming.get_shape().as_list()]


def dropout_selu(x, rate, alpha=-1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""
    
    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        
        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        
        if tensor_util.constant_value(keep_prob) == 1:
            return x
        
        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1 - binary_tensor)
        
        a = tf.sqrt(fixedPointVar / (keep_prob * ((1 - keep_prob) * tf.pow(alpha - fixedPointMean, 2) + fixedPointVar)))
        
        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret
    
    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
                                lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))


# ------------------------------------------------------------------------------------------------------------------
#  Classes
# ------------------------------------------------------------------------------------------------------------------
class Layer(object):
    def __init__(self):
        """Template class for all layers
        
        Parameters
        -------
        
        Returns
        -------
        
        Attributes
        -------
        .out : tensor or None
            Current output of the layer (does not trigger computation)
        """
        self.layer_scope = None
        self.out = None
    
    def get_output(self, **kwargs):
        """Calculate and return output of layer"""
        return self.out
    
    def get_output_shape(self):
        """Return shape of output (preferably without calculating the output)"""
        return []


class DropoutLayer(Layer):
    def __init__(self, incoming, prob, noise_shape=None, selu_dropout: bool = False, training: bool = True,
                 name='DropoutLayer'):
        """ Dropout layer using tensorflow dropout

        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            Incoming layer
        prob : float or False
            Probability to drop out an element
        noise_shape : list or None
            Taken from tensorflow documentation: By default, each element is kept or dropped independently. If
            noise_shape is specified, it must be broadcastable to the shape of x, and only dimensions with
            noise_shape[i] == shape(x)[i] will make independent decisions. For example, if shape(x) = [k, l, m, n] and
            noise_shape = [k, 1, 1, n], each batch and channel component will be kept independently and each row and
            column will be kept or not kept together.
            If None: drop out last dimension of input tensor consistently (i.e. drop out features);

        Returns
        -------
        """
        super(DropoutLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            if noise_shape is None:
                noise_shape = np.append(np.ones(len(self.incoming_shape) - 1, dtype=np.int32),
                                        [self.incoming_shape[-1]])
            else:
                self.noise_shape = noise_shape
            
            self.prob = prob
            self.noise_shape = noise_shape
            self.out = None
            self.name = name
            self.selu_dropout = selu_dropout
            self.training = training
    
    def get_output_shape(self):
        """Return shape of output"""
        return self.incoming_shape
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                if self.prob is not False:
                    if self.selu_dropout:
                        self.out = dropout_selu(incoming, rate=self.prob, noise_shape=self.noise_shape,
                                                training=self.training)
                    else:
                        self.out = tf.nn.dropout(incoming, keep_prob=1. - self.prob, noise_shape=self.noise_shape)
                else:
                    self.out = incoming
        
        return self.out


class DenseLayer(Layer):
    def __init__(self, incoming, n_units, flatten_input=False, W=tf.zeros, b=tf.zeros, a=tf.sigmoid, name='DenseLayer'):
        """ Dense layer, flexible enough to broadcast over time series

        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            Input of shape (samples, sequence_positions, features) or (samples, features) or (samples, ..., features);
        n_units : int
            Number of dense layer units
        flatten_input : bool
            True: flatten all inputs (samples[, ...], features) to shape (samples, -1); i.e. fully connect to everything
            per sample
            False: flatten all inputs (samples[, sequence_positions, ...], features) to shape
            (samples, sequence_positions, -1); i.e. fully connect to everything per sample or sequence position;
        W : initializer or tensor or tf.Variable
            Weights W either as initializer or tensor or tf.Variable; Will be used as learnable tf.Variable in any case;
        b : initializer or tensor or tf.Variable or None
            Biases b either as initializer or tensor or tf.Variable; Will be used as learnable tf.Variable in any case;
            No bias will be applied if b=None;
        a : function
            Activation function
        name : string
            Name of individual layer; Used as tensorflow scope;
            
        Returns
        -------
        """
        super(DenseLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            if (len(self.incoming_shape) > 2) and flatten_input:
                incoming_shape = [self.incoming_shape[0], np.prod(self.incoming_shape[1:])]
            elif len(self.incoming_shape) == 4:
                incoming_shape = [self.incoming_shape[0], np.prod(self.incoming_shape[1:])]
            elif len(self.incoming_shape) >= 5:
                incoming_shape = [self.incoming_shape[0], self.incoming_shape[1], np.prod(self.incoming_shape[2:])]
            else:
                incoming_shape = self.incoming_shape
            
            # Set init for W
            W = tofov(W, shape=[incoming_shape[-1], n_units], var_params=dict(name='W_dense'))
            
            # Set init for b
            if b is not None:
                b = tofov(b, [n_units], var_params=dict(name='b_dense'))
            
            self.a = a
            self.b = b
            self.W = W
            
            self.n_units = n_units
            self.flatten_input = flatten_input
            self.incoming_shape = incoming_shape
            
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        return [s if isinstance(s, int) and s >= 0 else -1 for s in self.incoming_shape[:-1]] + [self.n_units]
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer
        
        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                if (len(incoming.shape) > 2 and self.flatten_input) or (len(incoming.shape) > 3):
                    # Flatten all but first dimension (e.g. flat seq_pos and features)
                    X = tf.reshape(incoming, self.incoming_shape)
                else:
                    X = incoming
                net = dot_product(X, self.W)
                if self.b is not None:
                    net += self.b
                self.out = self.a(net)
        
        return self.out
    
    def get_weights(self):
        """Return list with all layer weights"""
        return [self.W]
    
    def get_biases(self):
        """Return list with all layer biases"""
        if self.b is None:
            return []
        else:
            return [self.b]


class ConvLayer(Layer):
    def __init__(self, incoming, W=None, b=tf.zeros, ksize: int = None, num_outputs: int = None,
                 weight_initializer=None, a=tf.nn.elu, strides=(1, 1, 1, 1), padding='ZEROPAD', dilation_rate=(1, 1),
                 name='ConvLayer'):
        """ Convolutional layer, flexible enough to broadcast over timeseries

        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            Input of shape (samples, sequence_positions, array_x, array_y, features) or
            (samples, array_x, array_y, features);
        W : tensorflow tensor or tf.Variable
            Initial value for weight kernel of shape (kernel_x, kernel_y, n_input_channels, n_output_channels)
        b : tensorflow initializer or tensor or tf.Variable or None
            Initial values or initializers for bias; None if no bias should be applied;
        ksize : int
            Kernel size; only used in conjunction with num_outputs and weight_initializer
        num_outputs : int
            Number of output feature maps; only used in conjunction with ksize and weight_initializer
        weight_initializer : initializer function
            Function for initialization of weight kernels; only used in conjunction with ksize and num_outputs
        a :  tensorflow function
            Activation functions for output
        strides : tuple
            Striding to use (see tensorflow convolution for further details)
        padding : str or tuple of int
            Padding method for image edges (see tensorflow convolution for further details); If specified as
            tuple or list of integer tf.pad is used to symmetrically zero-pad the x and y dimensions of the input.
            Furthermore supports TensorFlow paddings "VALID" and "SAME" in addition to "ZEROPAD" which symmetrically
            zero-pads the input so output-size = input-size / stride (taking into account strides and dilation;
            comparable to Caffe and Theano).
        dilation_rate : tuple of int or list of int
            Defaults to (1, 1) (i.e. normal 2D convolution). Use list of integers to specify multiple dilation rates;
            only for spatial dimensions -> len(dilation_rate) must be 2;
        
        Returns
        -------
        """
        super(ConvLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            # Set init for W and b
            if all(p is not None for p in [weight_initializer, ksize, num_outputs]):
                W = tofov(weight_initializer, shape=(ksize, ksize, self.incoming_shape[-1], num_outputs),
                          var_params=dict(name='W_conv'))
            else:
                W = tofov(W, shape=None, var_params=dict(name='W_conv'))
                ksize = W.get_shape()[0].value
            if b is not None:
                b = tofov(b, shape=W.get_shape().as_list()[-1], var_params=dict(name='b_conv'))
            
            self.a = a
            self.b = b
            self.W = W
            self.padding = padding
            self.strides = strides
            self.dilation_rate = dilation_rate
            
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        weights = self.W.get_shape().as_list()
        input_size = np.asarray(self.incoming_shape[-3:-1])
        strides = np.asarray(self.strides[-3:-1])
        kernels = np.asarray(weights[0:2])
        num_output = weights[-1]
        dilations = np.asarray(self.dilation_rate)
        if (isinstance(self.padding, list) or isinstance(self.padding, tuple)) and len(self.padding) == 2:
            output_size = np.asarray(
                np.ceil((input_size + 2 * np.asarray(self.padding) - kernels - (kernels - 1) * (dilations - 1)) / strides + 1),
                dtype=np.int)
        else:
            output_size = np.asarray(
                np.ceil(input_size / strides) if self.padding == "SAME" or self.padding == "ZEROPAD" else np.ceil(
                    (input_size - (kernels - 1) * dilations) / strides), dtype=np.int)
        
        output_shape = self.incoming_shape[:]
        output_shape[-3:-1] = output_size.tolist()
        output_shape[-1] = num_output
        return output_shape
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                # Perform convolution
                conv = conv2d(incoming, self.W, strides=self.strides, padding=self.padding,
                              dilation_rate=self.dilation_rate)
                
                # Add bias
                if self.b is not None:
                    conv += self.b
                
                # Apply activation function
                self.out = self.a(conv)
        
        return self.out
    
    def get_weights(self):
        """Return list with all layer weights"""
        return [self.W]
    
    def get_biases(self):
        """Return list with all layer biases"""
        if self.b is None:
            return []
        else:
            return [self.b]


class AvgPoolingLayer(Layer):
    def __init__(self, incoming, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME', data_format='NHWC',
                 name='MaxPoolingLayer'):
        """Average-pooling layer, capable of broadcasing over timeseries
        
        see tensorflow nn.avg_pool function for further details on parameters
        
        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            input to layer
            
        Returns
        -------
        """
        super(AvgPoolingLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            self.ksize = ksize
            self.strides = strides
            self.padding = padding
            self.data_format = data_format
            
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        input_size = np.asarray(self.incoming_shape[-3:-1] if self.data_format == "NHWC" else self.incoming_shape[-2:])
        strides = np.asarray(self.strides[-3:-1] if self.data_format == "NHWC" else self.strides[-2:])
        kernels = np.asarray(self.ksize[-3:-1] if self.data_format == "NHWC" else self.ksize[-2:])
        output_size = np.asarray(
            np.ceil(input_size / strides) if self.padding == "SAME" else np.ceil((input_size - (kernels - 1)) / strides), dtype=np.int)
        
        output_shape = self.incoming_shape[:]
        if self.data_format == "NHWC":
            output_shape[-3:-1] = output_size.tolist()
        else:
            output_shape[-2:] = output_size.tolist()
        
        return output_shape
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                self.out = avgpool2D(incoming, ksize=self.ksize, strides=self.strides, padding=self.padding,
                                     data_format=self.data_format)
        return self.out


class MaxPoolingLayer(Layer):
    def __init__(self, incoming, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME', data_format='NHWC',
                 name='MaxPoolingLayer'):
        """Max pooling layer, capable of broadcasting over time series
        
        see tensorflow max pooling function for further details on parameters
        
        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            input to layer
            
        Returns
        -------
        """
        super(MaxPoolingLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            self.ksize = ksize
            self.strides = strides
            self.padding = padding
            self.data_format = data_format
            
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        input_size = np.asarray(self.incoming_shape[-3:-1] if self.data_format == "NHWC" else self.incoming_shape[-2:])
        strides = np.asarray(self.strides[-3:-1] if self.data_format == "NHWC" else self.strides[-2:])
        kernels = np.asarray(self.ksize[-3:-1] if self.data_format == "NHWC" else self.ksize[-2:])
        output_size = np.asarray(
            np.ceil(input_size / strides) if self.padding == "SAME" else np.ceil((input_size - (kernels - 1)) / strides), dtype=np.int)
        
        output_shape = self.incoming_shape[:]
        if self.data_format == "NHWC":
            output_shape[-3:-1] = output_size.tolist()
        else:
            output_shape[-2:] = output_size.tolist()
        
        return output_shape
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                self.out = maxpool2D(incoming, ksize=self.ksize, strides=self.strides, padding=self.padding,
                                     data_format=self.data_format)
        return self.out


class ConcatLayer(Layer):
    def __init__(self, incomings, a=tf.identity, name='ConcatLayer'):
        """Concatenate outputs of multiple layers at last dimension (e.g. for skip-connections)
        
        Parameters
        -------
        incomings : list of layers, tensorflow tensors, or placeholders
            List of incoming layers to be concatenated
        
        Returns
        -------
        """
        super(ConcatLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incomings = []
            self.incoming_shapes = []
            
            for incoming in incomings:
                incoming, incoming_shape = get_input(incoming)
                self.incomings.append(incoming)
                self.incoming_shapes.append(incoming_shape)
            self.name = name
            self.a = a
    
    def get_output_shape(self):
        """Return shape of output"""
        return self.incoming_shapes[0][:-1] + [sum([s[-1] for s in self.incoming_shapes])]
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer
        
        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incomings = [incoming(prev_layers=prev_layers, **kwargs) for incoming in self.incomings]
            with tf.variable_scope(self.layer_scope):
                self.out = self.a(tf.concat(axis=len(self.incoming_shapes[0]) - 1, values=incomings))
        
        return self.out
