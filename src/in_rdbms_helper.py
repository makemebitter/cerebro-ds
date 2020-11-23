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

from imagenetcat import param_grid
from imagenetcat import SEED
from imagenetcat import INPUT_SHAPE
from criteocat import INPUT_SHAPE as INPUT_SHAPE_CRITEO
from criteocat import NUM_CLASSES as NUM_CLASSES_CRITEO
from imagenetcat import NUM_CLASSES
from imagenetcat import param_grid_hetro
from imagenetcat import param_grid_model_size
from imagenetcat import param_grid_best_model
from imagenetcat import param_grid_scalability
from criteocat import param_grid_criteo
from criteocat import param_grid_criteo_breakdown
from utils import set_seed
from utils import logs
import random
import argparse


def get_main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logs_root', type=str, default=''
    )
    parser.add_argument(
        '--models_root', type=str, default=''
    )
    parser.add_argument(
        '--train_name', type=str, default='imagenet_train_data_packed'
    )
    parser.add_argument(
        '--valid_name', type=str, default='imagenet_valid_data_packed'
    )
    parser.add_argument(
        '--run', action='store_true'
    )
    parser.add_argument(
        '--load', action='store_true'
    )
    parser.add_argument(
        '--db_name', type=str, default='cerebro'
    )
    parser.add_argument(
        '--size', type=int, default=8
    )
    parser.add_argument(
        '--num_epochs', type=int, default=10
    )
    parser.add_argument(
        '--drill_down_hetro', action='store_true'
    )
    parser.add_argument(
        '--drill_down_model_size', action='store_true'
    )
    parser.add_argument(
        '--drill_down_model_size_identifier', type=str, default='n'
    )
    parser.add_argument(
        '--drill_down_scalability', action='store_true'
    )
    parser.add_argument(
        '--best_model_run', action='store_true'
    )
    parser.add_argument(
        '--criteo', action='store_true'
    )
    parser.add_argument(
        '--criteo_breakdown', action='store_true'
    )
    return parser


def main_prepare(shuffle=True):
    parser = get_main_parser()
    args = parser.parse_args()
    logs("Size:{}".format(args.size))
    if args.size == 1:
        args.train_name = 'imagenet_train_data_packed_1'
        args.valid_name = 'imagenet_valid_data_packed_1'
        # args.train_name = 'imagenet_train_data1_packed'
        # args.valid_name = 'imagenet_valid_data1_packed'
    set_seed(SEED)
    msts = get_exp_specific_msts(args)
    if shuffle:
        random.shuffle(msts)
    logs(msts)
    return args, msts


def get_msts(param_grid=param_grid):
    param_names = list(param_grid.keys())

    def find_combinations(combinations, p, i):
        """

        :param combinations:
        :param p:
        :param i:
        """
        if i < len(param_names):
            for x in param_grid[param_names[i]]:
                p[param_names[i]] = x
                find_combinations(combinations, p, i + 1)
        else:
            combinations.append(p.copy())

    msts = []
    find_combinations(msts, {}, 0)
    msts = sorted(
        sorted(msts, key=lambda x: x['batch_size']), key=lambda x: x['model'])
    return msts


def get_exp_specific_msts(args):
    if args.criteo:
        if args.criteo_breakdown:
            msts = get_msts(param_grid=param_grid_criteo_breakdown)
        else:
            msts = get_msts(param_grid=param_grid_criteo)
    elif args.drill_down_hetro:
        msts = get_msts(param_grid=param_grid_hetro)
    elif args.drill_down_model_size:
        msts = get_msts(
            param_grid=param_grid_model_size[
                args.drill_down_model_size_identifier])
    elif args.best_model_run:
        msts = get_msts(
            param_grid=param_grid_best_model)
    elif args.drill_down_scalability:
        msts = get_msts(
            param_grid=param_grid_scalability)
    else:
        msts = get_msts()
    return msts


def mst_2_str(mst):
    return "learning_rate:{},lambda_value:{},batch_size:{},model:{}".format(
        mst['learning_rate'], mst['lambda_value'], mst['batch_size'],
        mst['model'])


def params_fac(mst):
    learning_rate = mst['learning_rate']
    batch_size = mst['batch_size']
    compile_params = "$$ loss='categorical_crossentropy', optimizer='Adam(lr={learning_rate})', metrics=['top_k_categorical_accuracy'] $$".format(
        **locals())
    fit_params = "$$ batch_size={batch_size}, epochs=1 $$".format(
        **locals())
    return compile_params, fit_params


def patch_model(model, lambda_value, seed=SEED, module=None):
    if not module:
        module = 'keras'
    if module == 'keras':
        import keras
    elif module == 'tf.keras':
        import tensorflow.keras as keras
    regularizer = keras.regularizers.l2(lambda_value)
    for layer in model.layers:
        for attr in ['kernel_regularizer', 'bias_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)
        for attr in ['kernel_initializer', 'bias_initializer']:
            if hasattr(layer, attr):
                layer_initializer = getattr(layer, attr)
                if hasattr(layer_initializer, 'seed'):
                    setattr(layer_initializer, 'seed', seed)
    return model


def create_model_from_mst(mst, module=None):
    if not module:
        module = 'keras'
    if module == 'keras':
        import keras
    elif module == 'tf.keras':
        import tensorflow.keras as keras
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
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(10, activation='relu', input_shape=(4,)))
        model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(3, activation='softmax'))
    elif mst['model'] == 'confA':
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(1000, activation='relu', input_shape=INPUT_SHAPE_CRITEO))
        model.add(keras.layers.Dense(500, activation='relu'))
        model.add(keras.layers.Dense(NUM_CLASSES_CRITEO, activation='softmax'))
    model = patch_model(model, mst['lambda_value'], module=module)
    return model
