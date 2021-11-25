# Adopted from https://github.com/Dawars/SqueezeNet-tf/blob/master/SqueezeNet.py
import tensorflow as tf


class SqueezeNet(object):

    def __init__(self, model_input, num_classes=1000, model_name='squeezenet'):
        self.imgs = model_input
        self.num_classes = num_classes
        self.model_name = model_name

        self.sq_ratio = 1.0

        self.__create()

    def __create(self):
        with tf.name_scope(self.model_name):
            # conv1_1
            conv1 = self.__conv_layer('conv1', self.imgs,
                                      W=self.__weight_variable([3, 3, 3, 64], name='conv1_w'), stride=[1, 2, 2, 1])

            relu1 = self.__relu_layer('relu1', conv1, b=self.__bias_variable([64], 'relu1_b'))
            pool1 = self.__pool_layer('pool1', relu1)

            fire2 = self.__fire_module('fire2', pool1, self.sq_ratio * 16, 64, 64)
            fire3 = self.__fire_module('fire3', fire2, self.sq_ratio * 16, 64, 64, True)
            pool3 = self.__pool_layer('pool3', fire3)

            fire4 = self.__fire_module('fire4', pool3, self.sq_ratio * 32, 128, 128)
            fire5 = self.__fire_module('fire5', fire4, self.sq_ratio * 32, 128, 128, True)
            pool5 = self.__pool_layer('pool5', fire5)

            fire6 = self.__fire_module('fire6', pool5, self.sq_ratio * 48, 192, 192)
            fire7 = self.__fire_module('fire7', fire6, self.sq_ratio * 48, 192, 192, True)
            fire8 = self.__fire_module('fire8', fire7, self.sq_ratio * 64, 256, 256)
            fire9 = self.__fire_module('fire9', fire8, self.sq_ratio * 64, 256, 256, True)

            conv10 = self.__conv_layer('conv10', fire9,
                                       W=self.__weight_variable([1, 1, 512, self.num_classes], name='conv10',
                                                                init='normal'))
            relu10 = self.__relu_layer('relu10', conv10, b=self.__bias_variable([self.num_classes], 'relu10_b'))
            pool10 = self.__pool_layer('pool10', relu10, pooling_type='avg')

            avg_pool_shape = tf.shape(pool10)
            self.logits = tf.reshape(pool10, [avg_pool_shape[0], -1])

    def __bias_variable(self, shape, name, value=0.1):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))

    def __weight_variable(self, shape, name=None, init='xavier'):
        if init == 'variance':
            initial = tf.get_variable('W' + name, shape, initializer=tf.contrib.layers.variance_scaling_initializer())
        elif init == 'xavier':
            initial = tf.get_variable('W' + name, shape, initializer=tf.contrib.layers.xavier_initializer())
        else:
            initial = tf.get_variable('W' + name, shape, initializer=tf.random_normal_initializer(stddev=0.01))

        return initial

    def __relu_layer(self, layer_name, layer_input, b=None):
        if b:
            layer_input += b
        relu = tf.nn.relu(layer_input)
        return relu

    def __pool_layer(self, layer_name, layer_input, pooling_type='max'):
        if pooling_type == 'avg':
            input_width = layer_input.get_shape()[1]
            pool = tf.nn.avg_pool(layer_input, ksize=[1, input_width, input_width, 1],
                                  strides=[1, 1, 1, 1], padding='VALID')
        elif pooling_type == 'max':
            pool = tf.nn.max_pool(layer_input, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1], padding='VALID')
        return pool

    def __conv_layer(self, layer_name, layer_input, W, stride=[1, 1, 1, 1]):
        return tf.nn.conv2d(layer_input, W, strides=stride, padding='SAME')

    def __fire_module(self, layer_name, layer_input, s1x1, e1x1, e3x3, residual=False):
        """ Fire module consists of squeeze and expand convolutional layers. """

        shape = layer_input.get_shape()

        # squeeze
        s1_weight = self.__weight_variable([1, 1, int(shape[3]), s1x1], layer_name + '_s1')

        # expand
        e1_weight = self.__weight_variable([1, 1, s1x1, e1x1], layer_name + '_e1')
        e3_weight = self.__weight_variable([3, 3, s1x1, e3x3], layer_name + '_e3')

        s1 = self.__conv_layer(layer_name + '_s1', layer_input, W=s1_weight)
        relu1 = self.__relu_layer(layer_name + '_relu1', s1,
                                  b=self.__bias_variable([s1x1], layer_name + '_fire_bias_s1'))

        e1 = self.__conv_layer(layer_name + '_e1', relu1, W=e1_weight)
        e3 = self.__conv_layer(layer_name + '_e3', relu1, W=e3_weight)
        concat = tf.concat([tf.add(e1, self.__bias_variable([e1x1],
                                                            name=layer_name + '_fire_bias_e1')),
                            tf.add(e3, self.__bias_variable([e3x3],
                                                            name=layer_name + '_fire_bias_e3'))], 3)

        if residual:
            relu2 = self.__relu_layer(layer_name + 'relu2_res', tf.add(concat, layer_input))
        else:
            relu2 = self.__relu_layer(layer_name + '_relu2', concat)

        return relu2
