import tensorflow as tf
from .cnn_utils import conv, fc, max_pool, avg_pool


class ResNet18(object):

    def __init__(self, model_input, num_classes=1000, model_name='resnet80'):
        self.model_input = model_input
        self.model_name = model_name
        self.num_classes = num_classes
        # Call the create function to build the computational graph of AlexNet
        self.__create()

    def __create(self):
        with tf.variable_scope(self.model_name):
            self.image = tf.cast(self.model_input, tf.float32)
            self.__calc_conv1()
            self.__calc_conv2()
            self.__calc_conv3()
            self.__calc_conv4()
            self.__calc_conv5()
            self.__calc_fc6()


    def __calc_conv1(self):
        temp = conv(self.image, 7, 7, 64, 2, 2, padding='VALID', name='conv1')
        self.conv1 = tf.nn.relu(temp)
        self.pool1 = max_pool(self.conv1, 3, 3, 2, 2, padding='SAME', name='pool1')

    def __calc_conv2(self):
        self.conv2_1 = tf.nn.relu(self.__conv_block(input_layer=self.pool1, name='2a',
                                                    num_filters=64, stride_x=1, stride_y=1))
        self.conv2_2 = tf.nn.relu(self.__identity_block(input_layer=self.conv2_1, name='2b',
                                                        num_filters=64))


    def __calc_conv3(self):
        self.conv3_1 = tf.nn.relu(self.__conv_block(input_layer=self.conv2_2, name='3a',
                                                    num_filters=128))
        self.conv3_2 = tf.nn.relu(self.__identity_block(input_layer=self.conv3_1, name='3b',
                                                        num_filters=128))

    def __calc_conv4(self):
        self.conv4_1 = tf.nn.relu(self.__conv_block(input_layer=self.conv3_2, name='4a',
                                                    num_filters=256))
        self.conv4_2 = tf.nn.relu(self.__identity_block(input_layer=self.conv4_1, name='4b',
                                                        num_filters=256))

    def __calc_conv5(self):
        self.conv5_1 = tf.nn.relu(self.__conv_block(input_layer=self.conv4_1, name='5a',
                                                    num_filters=512))
        self.conv5_2 = tf.nn.relu(self.__identity_block(input_layer=self.conv5_1, name='5b',
                                                        num_filters=512))

    def __calc_fc6(self):
        # 6th Layer: Flatten -> FC (w ReLu)
        width = self.conv5_2.get_shape()[1]
        self.pool_6 = avg_pool(self.conv5_2, width, width, width, width, padding='SAME', name='pool6')
        flattened = tf.reshape(self.pool_6, [-1, 1 * 1 * 2048])
        self.logits = fc(flattened, 1 * 1 * 2048, self.num_classes, name='fc')

    def __conv_block(self, input_layer, name, num_filters, stride_x=2, stride_y=2):

        with tf.name_scope('conv_block'):
            x = conv(input_layer, 1, 1, num_filters, 1, 1, padding='SAME', name='res' + name +
                                                '_branch2a')
            x = tf.nn.relu(x)

            x = conv(x, 3, 3, num_filters, stride_x, stride_y, padding='SAME', name='res' + name +
                                                '_branch2b')
            x = tf.nn.relu(x)

            x = conv(x, 1, 1, num_filters*4, 1, 1, padding='SAME', name='res' + name + '_branch2c')

            shortcut = conv(input_layer, 1, 1, num_filters*4, stride_x, stride_y, padding='SAME', name='res' + name + '_branch1')

            x = tf.add(x, shortcut)

        return x

    def __identity_block(self, input_layer, name, num_filters):

        with tf.name_scope('identity_block'):
            x = conv(input_layer, 1, 1, num_filters, 1, 1, padding='SAME', name='res' + name +
                                                '_branch2a')
            x = tf.nn.relu(x)

            x = conv(x, 3, 3, num_filters, 1, 1, padding='SAME', name='res' + name +
                                                '_branch2b')
            x = tf.nn.relu(x)

            x = conv(x, 1, 1, num_filters*4, 1, 1, padding='SAME', name='res' + name + '_branch2c')

            x = tf.add(x, input_layer)

        return x
