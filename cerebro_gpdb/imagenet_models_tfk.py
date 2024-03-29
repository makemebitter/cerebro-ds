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
import tensorflow.keras as keras

def create_model_from_mst(mst):
    if mst['model'] == 'vgg16':
        model = keras.applications.vgg16.VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'resnet50':
        model = keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'resnet152':
        model = keras.applications.ResNet152(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'densenet121':
        model = keras.applications.DenseNet121(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'nasnetmobile':
        model = keras.applications.NASNetMobile(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'mobilenetv2':
        model = keras.applications.MobileNetV2(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'mobilenetv1':
        model = keras.applications.MobileNet(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'sanity':
        model = Sequential()
        model.add(Dense(10, activation='relu', input_shape=(4,)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(3, activation='softmax'))
    model = patch_model(model, mst['lambda_value'])
    return model