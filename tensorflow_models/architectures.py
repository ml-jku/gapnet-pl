import tensorflow as tf

from TeLL.activations import selu
from TeLL.config import Config
from TeLL.initializations import weight_xavier_conv2d, scaled_elu_initialization
from TeLL.layers import DenseLayer, ConvLayer, MaxPoolingLayer, DropoutLayer, ConcatLayer, \
    AvgPoolingLayer


class DeepLoc(object):
    def __init__(self, config: Config, shape):
        width = shape[1]
        height = shape[0]

        n_classes = config.get_value("n_classes", 13)
        n_channels = config.get_value("n_channels", 4)

        act = selu
        w_init = scaled_elu_initialization

        # tf Graph input
        X = tf.placeholder(tf.float32, [None, height, width, n_channels], name="Features")
        y_ = tf.placeholder(tf.float32, [None, n_classes], name="Labels")
        d = tf.placeholder(tf.float32)

        print(X.get_shape())

        layers = list()
        layers.append(conv(X, w_init, act, k=3, s=1, out=32, id=1))
        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=32, id=2))
        layers.append(maxpool(layers[-1], k=2, s=2, id=1))

        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=64, id=3))
        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=64, id=4))
        layers.append(maxpool(layers[-1], k=2, s=2, id=2))

        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=96, id=5))
        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=96, id=6))
        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=96, id=7))
        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=96, id=8))
        layers.append(maxpool(layers[-1], k=2, s=2, id=3))

        # dense layers
        layers.append(fc(layers[-1], w_init, act, units=128, flatten=True, id=1))
        layers.append(dropout(layers[-1], d, act, id=1))

        layers.append(fc(layers[-1], w_init, act, units=128, flatten=False, id=2))
        layers.append(dropout(layers[-1], d, act, id=2))

        layers.append(fc(layers[-1], w_init, tf.identity, units=n_classes, flatten=False, id=3))

        # publish
        self.X = X
        self.y_ = y_
        self.dropout = d
        self.output = layers[-1].get_output()
        self.layers = layers


class GapNetPL(object):
    def __init__(self, config: Config, shape):
        width = shape[1]
        height = shape[0]

        output_units = config.get_value("n_classes", 13)
        n_channels = config.get_value("n_channels", 4)

        activation = selu
        weight_init = scaled_elu_initialization

        n_full1 = config.get_value("n_full1", 256)
        n_full2 = config.get_value("n_full1", 256)

        # tf Graph input, compatible to DenseNet
        X = tf.placeholder(tf.float32, [None, height, width, n_channels], name="Features")
        y_ = tf.placeholder(tf.float32, [None, output_units], name="Labels")
        d = tf.placeholder(tf.float32)

        layers = list()
        layers.append(conv(X, weight_init, activation, k=3, s=2, out=32, id=1))
        layers.append(maxpool(layers[-1], k=2, s=2, id=1))
        blk1 = layers[-1]
        blk1_bmp = layers[-2]
        print("Block 1: {}".format(blk1.get_output_shape()))

        layers.append(conv(layers[-1], weight_init, activation, k=3, s=2, out=64, id=2))
        layers.append(conv(layers[-1], weight_init, activation, k=3, s=1, out=64, id=3))
        layers.append(conv(layers[-1], weight_init, activation, k=3, s=1, out=64, id=4))
        layers.append(conv(layers[-1], weight_init, activation, k=3, s=1, out=64, id=5))
        layers.append(maxpool(layers[-1], k=2, s=2, id=2))
        blk2 = layers[-1]
        blk2_bmp = layers[-2]
        print("Block 2: {}".format(blk2.get_output_shape()))

        layers.append(conv(layers[-1], weight_init, activation, k=3, s=1, out=128, id=6))
        layers.append(conv(layers[-1], weight_init, activation, k=3, s=1, out=128, id=7))
        layers.append(conv(layers[-1], weight_init, activation, k=3, s=1, out=128, id=8))
        blk3 = layers[-1]
        print("Block 3: {}".format(blk3.get_output_shape()))

        # global average pooling
        layers.append(global_average(blk1, id=1))
        layers.append(global_average(blk2, id=2))
        layers.append(global_average(blk3, id=3))

        # concat
        layers.append(ConcatLayer(layers[-3:], name="ConcatAverage"))

        print("Concat: {}".format(layers[-1].get_output_shape()))

        # FC
        layers.append(fc(layers[-1], weight_init, activation, n_full1, flatten=True, id=1))
        layers.append(dropout(layers[-1], d, activation, 2))
        print("FC 1: {}".format(layers[-1].get_output_shape()))

        layers.append(fc(layers[-1], weight_init, activation, n_full2, flatten=False, id=2))
        layers.append(dropout(layers[-1], d, activation, 3))
        print("FC 2: {}".format(layers[-1].get_output_shape()))

        layers.append(fc(layers[-1], weight_init, tf.identity, output_units, flatten=False, id=3))

        # publish
        self.X = X
        self.y_ = y_
        self.dropout = d
        self.blk1_bmp = blk1_bmp
        self.blk2_bmp = blk2_bmp
        self.blk3 = blk3
        self.fc1 = layers[-5]
        self.fc2 = layers[-3]
        self.out = layers[-1]
        self.layers = layers
        self.output = layers[-1].get_output()


class MIL(object):
    def __init__(self, config: Config, shape):
        width = shape[1]
        height = shape[0]

        n_classes = config.get_value("n_classes", 13)
        a = config.get_value('a', 5)

        act = tf.nn.relu
        w_init = weight_xavier_conv2d

        # tf Graph input
        X = tf.placeholder(tf.float32, [None, height, width, 4], name="Features")
        y_ = tf.placeholder(tf.float32, [None, n_classes], name="Labels")
        # dropout
        d = tf.placeholder(tf.float32)
        d1 = tf.cond(tf.equal(d, tf.constant(0, dtype=tf.float32)), lambda: tf.constant(0, dtype=tf.float32),
                     lambda: tf.constant(0.2, dtype=tf.float32))
        d2 = tf.cond(tf.equal(d, tf.constant(0, dtype=tf.float32)), lambda: tf.constant(0, dtype=tf.float32),
                     lambda: tf.constant(0.5, dtype=tf.float32))

        print(X.get_shape())

        layers = list()
        layers.append(avgpool(X, k=3, s=2, id=1))
        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=32, id=1))
        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=64, id=2))
        layers.append(maxpool(layers[-1], k=3, s=2, id=1))
        layers.append(dropout(layers[-1], d1, act, id=1))

        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=64, id=3))
        layers.append(maxpool(layers[-1], k=3, s=2, id=2))
        layers.append(dropout(layers[-1], d1, act, id=2))

        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=128, id=4))
        layers.append(maxpool(layers[-1], k=3, s=2, id=3))
        layers.append(dropout(layers[-1], d1, act, id=3))

        layers.append(conv(layers[-1], w_init, act, k=3, s=1, out=128, id=5))
        layers.append(maxpool(layers[-1], k=3, s=2, id=4))
        layers.append(dropout(layers[-1], d1, act, id=4))

        layers.append(conv(layers[-1], w_init, act, k=1, s=1, out=1000, id=6))
        layers.append(dropout(layers[-1], d2, act, id=5))

        # intermediate output layer
        layers.append(conv(layers[-1], w_init, tf.identity, k=1, s=1, out=n_classes, id=7))
        # noisyAnd pooling
        with tf.variable_scope('NoisyAND'):
            a = tf.get_variable(name='a', shape=[1], initializer=tf.constant_initializer(a), trainable=False)
            b = tf.get_variable(name='b', shape=[1, n_classes], initializer=tf.constant_initializer(0.0))
            b = tf.clip_by_value(b, 0.0, 1.0)
            mean = tf.reduce_mean(tf.nn.sigmoid(layers[-1].get_output()), axis=[1, 2])
            noisyAnd = (tf.nn.sigmoid(a * (mean - b)) - tf.nn.sigmoid(-a * b)) / \
                       (tf.sigmoid(a * (1 - b)) - tf.sigmoid(-a * b))

        # output layer
        layers.append(
            fc(layers[-1], tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32),
               tf.identity, units=n_classes, flatten=False, id=1))

        # publish
        self.X = X
        self.y_ = y_
        self.dropout = d
        self.output_nand = noisyAnd
        self.output = layers[-1].get_output(prev_layers=[layers[-2]])
        self.layers = layers

    def loss(self):
        # loss of output layer
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.y_))
        # loss of mil pooling layer
        mil = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_nand, labels=self.y_))
        return loss + mil


def getPatchInput(config, X):
    patches = tf.extract_image_patches(images=X,
                                       ksizes=[1, config.patch_size, config.patch_size, 1],
                                       strides=[1, config.patch_size, config.patch_size, 1],
                                       rates=[1, 1, 1, 1], padding='VALID')

    input = tf.reshape(patches, [-1, config.patch_size, config.patch_size, config.get_value("n_channels", 4)])

    patchesPerImage = tf.cast((tf.shape(input)[0]) / (tf.shape(X)[0]), tf.int32)
    segment_ids = tf.reshape(tf.tile(tf.expand_dims(tf.range(config.batchsize), -1), [1, patchesPerImage]), [-1])
    return input, segment_ids, patchesPerImage


def conv(input, init, act, k, s, out, id, drop_rate=0., dilation=1):
    return ConvLayer(input, weight_initializer=init, ksize=k, num_outputs=out, name="ConvLayer-{}".format(id),
                     a=act, strides=(1, s, s, 1), dilation_rate=(dilation, dilation))


def fc(input, init, act, units, flatten, id):
    return DenseLayer(input, units, name="FC-{}".format(id), a=act, W=init, b=tf.zeros, flatten_input=flatten)


def dropout(input, prob, act, id):
    return DropoutLayer(input, prob=prob, selu_dropout=True if act.__name__ == "selu" else False,
                        name="Dropout-{}".format(id))


def maxpool(input, k, s, id, padding="SAME"):
    return MaxPoolingLayer(input, name="MaxPoolingLayer-{}".format(id), ksize=(1, k, k, 1), strides=(1, s, s, 1),
                           padding=padding)


def avgpool(input, k, s, id, padding="SAME"):
    return AvgPoolingLayer(input, name="AvgPoolingLayer-{}".format(id), ksize=(1, k, k, 1), strides=(1, s, s, 1),
                           padding=padding)


def global_average(input, id):
    input_shape = input.get_output_shape()
    return AvgPoolingLayer(input, ksize=(1, input_shape[1], input_shape[2], 1), padding="VALID",
                           name="GlobalAvgPooling-{}".format(id))
