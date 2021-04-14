# Copyright 2020 Yuhao Zhang and Arun Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from resnet50tfk import KI

preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
layers = tf.keras.layers
models = tf.keras.models
keras_utils = tf.keras.utils


def VGG16(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          lambda_value=None,
          **kwargs):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    KR = tf.keras.regularizers.l2(lambda_value)
    global layers, models, keras_utils
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    # # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=32,
    #                                   data_format=backend.image_data_format(),
    #                                   require_flatten=include_top,
    #                                   weights=weights)

    # if input_tensor is None:
    #     img_input = layers.Input(shape=input_shape)
    # else:
    #     if not backend.is_keras_tensor(input_tensor):
    #         img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    product = 1
    for x in input_shape:
        product *= x
    img_input = layers.Input(shape=(product, ))
    x = layers.Reshape(input_shape)(img_input)

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block1_conv1')(x)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(
        2, 2), name='block1_pool', padding='same')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(
        2, 2), name='block2_pool', padding='same')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(
        2, 2), name='block3_pool', padding='same')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(
        2, 2), name='block4_pool', padding='same')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=KI,
                      bias_initializer=KI,
                      kernel_regularizer=KR,
                      bias_regularizer=KR,
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(
        2, 2), name='block5_pool', padding='same')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', kernel_initializer=KI,
                         bias_initializer=KI,
                         kernel_regularizer=KR,
                         bias_regularizer=KR, name='fc1')(x)
        x = layers.Dense(4096, activation='relu', kernel_initializer=KI,
                         bias_initializer=KI,
                         kernel_regularizer=KR,
                         bias_regularizer=KR, name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', kernel_initializer=KI,
                         bias_initializer=KI,
                         kernel_regularizer=KR,
                         bias_regularizer=KR, name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='vgg16')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
