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
import h5py
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from cerebro_gpdb.imagenetcat import NUM_CLASSES
from cerebro_gpdb.imagenetcat import INPUT_SHAPE
from cerebro_gpdb.imagenetcat import SEED
from cerebro_gpdb.run_cerebro_standalone_helper import input_fn
import gc


def dataset_patch(dataset, mst, epochs):
    # 
    return dataset.shuffle(1000, seed=SEED).prefetch(
        tf.data.experimental.AUTOTUNE).batch(
        mst['batch_size']).repeat(count=epochs)


class generator_h5:
    def __init__(self, filelist):
        self.filelist = filelist

    def __call__(self):
        for filename in self.filelist:
            # self.data = input_fn(filename)
            # gc.collect()
            # for x in self.data:
            #     for i in range(x["images"].shape[0]):
            #         image_arr = x["images"][i]
            #         label = x["labels"][i]
            #         yield image_arr, label
            with h5py.File(filename, 'r') as hf:
                length = hf["images"].shape[0]
                for i in range(length):
                    image = hf["images"][i]
                    label = int(hf["labels"][i])
                    label = np.eye(NUM_CLASSES, dtype=np.float32)[label]
                    yield image, label


def data_h5(epochs, mst, train_list=None, valid_list=None):
    if not train_list:
        train_list = sorted(glob.glob('/mnt/imagenet/train/*.h5'))
    if not valid_list:
        valid_list = sorted(glob.glob('/mnt/imagenet/valid/*.h5'))

    print(train_list, valid_list)

    def get_len(filelist):
        count = 0
        for filename in filelist:
            with h5py.File(filename, 'r') as hf:
                length = hf["images"].shape[0]
            count += length
        return count

    train_len = get_len(train_list)
    valid_len = get_len(valid_list)
    train_steps = train_len // mst['batch_size']
    valid_steps = valid_len // mst['batch_size']
    train_dataset = tf.data.Dataset.from_generator(
        generator_h5(train_list),
        output_types=(tf.float32, tf.int16),
        output_shapes=(INPUT_SHAPE, NUM_CLASSES))
    train_dataset = dataset_patch(train_dataset, mst, epochs)

    valid_dataset = tf.data.Dataset.from_generator(
        generator_h5(valid_list),
        output_types=(tf.float32, tf.int16),
        output_shapes=(INPUT_SHAPE, NUM_CLASSES))
    valid_dataset = dataset_patch(valid_dataset, mst, epochs)
    return train_dataset, valid_dataset, train_steps, valid_steps


def create_optimizer_from_mst(mst):
    optimizer = keras.optimizers.Adam(lr=mst['learning_rate'])
    return optimizer


def compile_model_from_mst(mst, model):
    optimizer = create_optimizer_from_mst(mst)
    top_5 = 'top_k_categorical_accuracy'
    top_1 = 'categorical_accuracy'
    model.compile(optimizer=optimizer,  # Optimizer
                  # Loss function to minimize
                  loss='categorical_crossentropy',
                  # List of metrics to monitor
                  metrics=[top_5, top_1])
    return model


class RefreshOptimizer(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, mst):
        super(RefreshOptimizer, self).__init__()
        self.mst = mst

    def on_epoch_begin(self, epoch, logs=None):
        # recompile
        print("Recompiling model")
        compile_model_from_mst(self.mst, self.model)
