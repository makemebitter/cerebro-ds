# Adopted from https://github.com/taki0112/Densenet-Tensorflow

import tensorflow as tf


class DenseNet(object):

    def __init__(self, model_input, num_classes=1000, model_name='densenet'):
        self.model_input = model_input
        self.num_classes = num_classes
        self.model_name = model_name
        self.nb_blocks = 2
        self.filters = 24
        self.__create()

    def __create(self):
        with tf.name_scope(self.model_name):
            self.image = tf.cast(self.model_input, tf.float32)
            x = self.__conv_layer(self.image, filter=2 * self.filters, kernel=[7, 7], stride=2, layer_name='conv0')

            x = self.__dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
            x = self.__transition_layer(x, scope='trans_1')

            x = self.__dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
            x = self.__transition_layer(x, scope='trans_2')

            x = self.__dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
            x = self.__transition_layer(x, scope='trans_3')

            x = self.__dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

            x = tf.nn.relu(x)
            input_width = x.get_shape()[1]
            x = tf.layers.average_pooling2d(inputs=x, pool_size=[input_width, input_width], strides=1, padding='VALID')
            flattened = tf.reshape(x, [-1, 1 * 1 * x.get_shape()[-1]])
            self.logits = tf.layers.dense(inputs=flattened, units=self.num_classes, name='logits')

    def __bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = tf.nn.relu(x)
            x = self.__conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')

            x = tf.nn.relu(x)
            x = self.__conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope + '_conv2')

            return x

    def __transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = tf.nn.relu(x)
            x = self.__conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = tf.layers.average_pooling2d(inputs=x, pool_size=[2, 2], strides=2, padding='VALID')
            return x

    def __dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.__bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = tf.concat(layers_concat, axis=3)
                x = self.__bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = tf.concat(layers_concat, axis=3)

            return x

    def __conv_layer(self, input, filter, kernel, stride=1, layer_name="conv"):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                       padding='SAME')
            return network
