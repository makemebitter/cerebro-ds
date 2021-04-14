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
import os
import argparse
import tensorflow as tf
from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator
from cerebro.storage import HDFSStore
from cerebro.tune import GridSearch, hp_choice
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf

from imagenetcat import NUM_CLASSES
from imagenetcat import INPUT_SHAPE
import resnet50tfk_default
import vgg16tfk_default
import resnet50tfk
import vgg16tfk
import dill
from imagenetcat import param_grid
from cerebro_gpdb.imagenetcat import spark_imagenet_cat
from cerebro_gpdb.cerebro_spark_wrapper import CerebroSparkImageNetBase
from utils import logs

os.environ["PYSPARK_PYTHON"] = "/mnt/py3v/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/mnt/py3v/bin/python"





# For sanity checks
# class spark_imagenet_cat:
#     valid_list = [
#         "hdfs://master:9000/imagenet_parquet/valid/valid_0.parquet"]
#     train_list = [
#         "hdfs://master:9000/imagenet_parquet/valid/valid_0.parquet"]

# param_grid = {
#     "learning_rate": [1e-4, 1e-6],
#     "lambda_value": [1e-4, 1e-6],
#     "batch_size": [32, 256],
#     "model": ["resnet50"]
# }

# param_grid = {
#         "learning_rate": [1e-4, 1e-6],
#         "lambda_value": [1e-4, 1e-6],
#         "batch_size": [32, 256],
#         "model": ["vgg16", "resnet50"]
#     }


def estimator_gen_fn(params):
    is_keras_default = False
    lambda_value = params['lambda_value']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    model_name = params['model']
    # For compliance with Reduction.SUM
    # lambda_value = lambda_value / batch_size
    if is_keras_default:
        product = 1
        for x in INPUT_SHAPE:
            product *= x
        input_vec = tf.keras.layers.Input(shape=(product,))
        input_tensor = tf.keras.layers.Reshape(INPUT_SHAPE)(input_vec)
        KR = tf.keras.regularizers.l2(lambda_value)
        if model_name == 'resnet50':
            model = resnet50tfk_default.ResNet50(include_top=True,
                                                 weights=None,
                                                 input_tensor=input_tensor,
                                                 input_shape=INPUT_SHAPE,
                                                 pooling=None,
                                                 classes=NUM_CLASSES,
                                                 KR=KR)
            loss = tf.keras.losses.CategoricalCrossentropy()
        if model_name == 'vgg16':
            model = vgg16tfk_default.VGG16(include_top=True,
                                           weights=None,
                                           input_tensor=input_tensor,
                                           input_shape=INPUT_SHAPE,
                                           pooling=None,
                                           classes=NUM_CLASSES,
                                           KR=KR)
            loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        if model_name == 'resnet50':
            model = resnet50tfk.ResNet50(include_top=True,
                                         weights=None,
                                         input_tensor=None,
                                         input_shape=INPUT_SHAPE,
                                         pooling=None,
                                         classes=NUM_CLASSES,
                                         lambda_value=lambda_value)
        elif model_name == 'vgg16':
            model = vgg16tfk.VGG16(include_top=True,
                                   weights=None,
                                   input_tensor=None,
                                   input_shape=INPUT_SHAPE,
                                   pooling=None,
                                   classes=NUM_CLASSES,
                                   lambda_value=lambda_value)
        loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    top_5 = 'top_k_categorical_accuracy'
    top_1 = 'categorical_accuracy'
    keras_estimator = SparkEstimator(model=model,
                                     optimizer=optimizer,
                                     loss=loss,
                                     metrics=[top_5],
                                     batch_size=batch_size)
    return keras_estimator


# defined only for back-compatibility.
# Use cerebro_spark_wrapper.CerebroSparkImageNet instead
class CerebroSparkImageNet(CerebroSparkImageNetBase):
    def run(self, spark_imagenet_cat, epoch=10, prepared=False):
        if prepared:
            logs("Skipping data loading")
        else:
            logs("Starting data loading")
            df = self.load(spark_imagenet_cat)
            logs("Ending data loading")
        self.search_space = {key: hp_choice(value)
                             for key, value in param_grid.items()}
        self.grid_search = GridSearch(self.backend,
                                      self.store,
                                      estimator_gen_fn,
                                      self.search_space,
                                      epoch,
                                      validation="validation",
                                      evaluation_metric='loss',
                                      feature_columns=['features'],
                                      label_columns=['labels'],
                                      verbose=2)
        print("param_maps:{}".format(self.grid_search.estimator_param_maps))
        if prepared:
            logs("Skipping data preparing")
        else:
            logs("Starting data preparing")
            self.backend.prepare_data(
                self.store, df, validation='validation', feature_columns=[
                    'features'], label_columns=['labels'])
            logs("Ending data preparing")

        logs("Starting training")
        model = self.grid_search.fit_on_prepared_data()
        logs("Ending training")
        return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outroot', type=str, default='/mnt/nfs'
    )
    parser.add_argument(
        '--run', action='store_true'
    )
    parser.add_argument(
        '--prepared', action='store_true'
    )
    parser.add_argument(
        '--epoch', type=int, default=10
    )
    args = parser.parse_args()

    if args.run:
        logs("START RUNNING")
        runner = CerebroSparkImageNet(8)
        model = runner.run(spark_imagenet_cat, args.epoch, args.prepared)
        print("best_model_history:{}".format(model.get_best_model_history()))
        print("all_model_history:{}".format(model.get_all_model_history()))
        with open(os.path.join(args.outroot, 'model_params.dill'), "wb") as f:
            dill.dump(runner.grid_search.estimator_param_maps, f)
        with open(os.path.join(args.outroot, 'history.dill'), "wb") as f:
            dill.dump(model.get_all_model_history(), f)
        logs("END RUNNING")
