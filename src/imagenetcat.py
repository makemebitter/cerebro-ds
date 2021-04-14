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
import glob
SEED = 2018
INPUT_SHAPE = (112, 112, 3)
NUM_CLASSES = 1000
TOP_5 = 'top_k_categorical_accuracy'
TOP_1 = 'categorical_accuracy'
MODEL_ARCH_TABLE = 'model_arch_library'
MODEL_SELECTION_TABLE = 'mst_table'
MODEL_SELECTION_SUMMARY_TABLE = 'mst_table_summary'


class spark_imagenet_cat:
    valid_list = [
        "hdfs://master:9000/imagenet_parquet/valid/valid_{}.parquet".format(i) for i in range(8)]
    train_list = [
        "hdfs://master:9000/imagenet_parquet/train/train_{}.parquet".format(i) for i in range(8)]


class spark_imagenet_cat_nfs:
    valid_list = [
        "/mnt/nfs/hdd/imagenet/valid/valid_{}.parquet".format(i) 
        for i in range(8)
    ]
    train_list = [
        "/mnt/nfs/hdd/imagenet/train/train_{}.parquet".format(i) 
        for i in range(8)
    ]


param_grid = {
    "learning_rate": [1e-4, 1e-6],
    "lambda_value": [1e-4, 1e-6],
    "batch_size": [32, 256],
    "model": ["vgg16", "resnet50"]
}
param_grid_hetro = {
    "learning_rate": [1e-4, 1e-4],
    "lambda_value": [1e-4, 1e-4],
    "batch_size": [4, 128],
    "model": ["nasnetmobile", "mobilenetv2"],
    'p': 0.8,
    'hetro': True,
    'fast': 38,
    'slow': 10,
    'total': 48
}

param_grid_scalability = {
    "learning_rate": [1e-3, 1e-4, 1e-5, 1e-6],
    "lambda_value": [1e-4, 1e-6],
    "batch_size": [32],
    "model": ["resnet50"]
}
param_grid_model_size = {
    's': {
        "learning_rate": [1e-4, 1e-6],
        "lambda_value": [1e-3, 1e-4, 1e-5, 1e-6],
        "batch_size": [32],
        "model": ["mobilenetv2"]
    },
    'm': {
        "learning_rate": [1e-4, 1e-6],
        "lambda_value": [1e-3, 1e-4, 1e-5, 1e-6],
        "batch_size": [32],
        "model": ["resnet50"]
    },
    'l': {
        "learning_rate": [1e-4, 1e-6],
        "lambda_value": [1e-3, 1e-4, 1e-5, 1e-6],
        "batch_size": [32],
        "model": ["resnet152"]
    },
    'x': {
        "learning_rate": [1e-4, 1e-6],
        "lambda_value": [1e-3, 1e-4, 1e-5, 1e-6],
        "batch_size": [32],
        "model": ["vgg16"]
    },
}
param_grid_best_model = {
    "learning_rate": [1e-4],
    "lambda_value": [1e-4],
    "batch_size": [32],
    "model": ["resnet50"]
}
param_grid_hyperopt = {
    "learning_rate": [0.00001, 0.1],
    "lambda_value": [1e-4, 1e-6],
    "batch_size": [16, 256],
    "model": ["resnet18", "resnet34"]
}
