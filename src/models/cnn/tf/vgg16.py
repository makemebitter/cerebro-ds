'''
Copyright 2018 Supun Nakandala and Arun Kumar
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import os

import tensorflow as tf

from .cnn_utils import conv, fc, max_pool


class VGG16(object):

    def __init__(self, model_input, num_classes=1000, model_name='vgg16'):
        self.model_input = model_input
        self.model_name = model_name
        self.num_classes = num_classes

        # Call the create function to build the computational graph of AlexNet
        self.__create()

    def __create(self):
        with tf.name_scope(self.model_name):
            self.image = tf.cast(self.model_input, tf.float32)
            self.__calc_conv1()
            self.__calc_conv2()
            self.__calc_conv3()
            self.__calc_conv4()
            self.__calc_conv5()
            self.__calc_fc6()
            self.__calc_fc7()
            self.__calc_fc8()

    def __calc_conv1(self):
        self.conv1_1 = tf.nn.relu(conv(self.image, 3, 3, 64, 1, 1, name='conv1_1',
                                        padding='SAME'))
        self.conv1_2 = tf.nn.relu(conv(self.conv1_1, 3, 3, 64, 1, 1, name='conv1_2',
                                        padding='SAME'))
        self.pool1 = max_pool(self.conv1_2, 2, 2, 2, 2, name='pool1', padding='SAME')

    def __calc_conv2(self):
        self.conv2_1 = tf.nn.relu(conv(self.pool1, 3, 3, 128, 1, 1, name='conv2_1'))
        self.conv2_2 = tf.nn.relu(conv(self.conv2_1, 3, 3, 128, 1, 1, name='conv2_2'))
        self.pool2 = max_pool(self.conv2_2, 2, 2, 2, 2, name='pool2')

    def __calc_conv3(self):
        self.conv3_1 = tf.nn.relu(conv(self.pool2, 3, 3, 256, 1, 1, name='conv3_1'))
        self.conv3_2 = tf.nn.relu(conv(self.conv3_1, 3, 3, 256, 1, 1, name='conv3_2'))
        self.conv3_3 = tf.nn.relu(conv(self.conv3_2, 3, 3, 256, 1, 1, name='conv3_3'))
        self.pool3 = max_pool(self.conv3_3, 2, 2, 2, 2, name='pool3')

    def __calc_conv4(self):
        self.conv4_1 = tf.nn.relu(conv(self.pool3, 3, 3, 512, 1, 1, name='conv4_1'))
        self.conv4_2 = tf.nn.relu(conv(self.conv4_1, 3, 3, 512, 1, 1, name='conv4_2'))
        self.conv4_3 = tf.nn.relu(conv(self.conv4_2, 3, 3, 512, 1, 1, name='conv4_3'))
        self.pool4 = max_pool(self.conv4_3, 2, 2, 2, 2, name='pool4')

    def __calc_conv5(self):
        self.conv5_1 = tf.nn.relu(conv(self.pool4, 3, 3, 512, 1, 1, name='conv5_1'))
        self.conv5_2 = tf.nn.relu(conv(self.conv5_1, 3, 3, 512, 1, 1, name='conv5_2'))
        self.conv5_3 = tf.nn.relu(conv(self.conv5_2, 3, 3, 512, 1, 1, name='conv5_3'))
        self.pool5 = max_pool(self.conv5_3, 2, 2, 2, 2, name='pool5')

    def __calc_fc6(self):
        width = self.pool5.shape[1]
        flattened = tf.reshape(self.pool5, [-1, width * width * 512])
        self.fc6 = tf.nn.relu(fc(flattened, width * width * 512, 4096, name='fc6'))

    def __calc_fc7(self):
        self.fc7 = tf.nn.relu(fc(self.fc6, 4096, 4096, name='fc7'))

    def __calc_fc8(self):
        self.logits = fc(self.fc7, 4096, self.num_classes, name='fc8')
