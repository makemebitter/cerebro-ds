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

import tensorflow as tf
import pathmagic # noqa
from cerebro_gpdb.imagenetcat import TOP_5
from cerebro_gpdb.imagenetcat import TOP_1
from cerebro_gpdb.imagenetcat import INPUT_SHAPE
from cerebro_gpdb.imagenetcat import NUM_CLASSES
from cerebro_gpdb.criteocat import INPUT_SHAPE as INPUT_SHAPE_CRITEO
from cerebro_gpdb.criteocat import NUM_CLASSES as NUM_CLASSES_CRITEO
from cerebro_gpdb.run_imagenet import create_model_from_mst
import argparse

def get_dataset(data, mst, generator_data):
    gen = generator_data(data)
    num_steps = gen.length // mst['batch_size']
    if mst['model'] == 'confA':
        input_shape = INPUT_SHAPE_CRITEO 
        num_classes = NUM_CLASSES_CRITEO
    else:
        input_shape = INPUT_SHAPE 
        num_classes = NUM_CLASSES
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.int16),
        output_shapes=(
            input_shape,
            num_classes
        )
    ).prefetch(tf.data.experimental.AUTOTUNE).batch(mst['batch_size'])
    return dataset, num_steps


def get_model(mst, module=None):
    if not module:
        module = 'keras'
    if module == 'keras':
        import keras
    elif module == 'tf.keras':
        import tensorflow.keras as keras
    model = create_model_from_mst(mst, module=module)
    optimizer = keras.optimizers.Adam(lr=mst['learning_rate'])
    model.compile(optimizer=optimizer,  # Optimizer
                  # Loss function to minimize
                  loss='categorical_crossentropy',
                  # List of metrics to monitor
                  metrics=[TOP_5, TOP_1])
    return model


def model_fn(generator_data):
    def model_fn_closure(data, mst, purpose):
        """

        :param data:
        :param mst:
        :return:
        """
        model = None
        dataset = None
        num_steps = None
        if purpose == 'data' or purpose == 'both':
            dataset, num_steps = get_dataset(data, mst, generator_data)
        if purpose == 'model' or purpose == 'both':
            model = get_model(mst, module='tf.keras')
        return dataset, num_steps, model
    return model_fn_closure


def train_fn(model, dataset, num_steps, epoch, train=True):
    if train:
        history = model.fit(x=dataset,
                            epochs=epoch+1,
                            verbose=1,
                            initial_epoch=epoch,
                            steps_per_epoch=num_steps,
                            validation_split=0
                            )
        res = history.history
        res = {k: v[-1] for k, v in res.items()}
    else:
        res = model.evaluate(dataset, steps=num_steps)
        res = dict(zip(model.metrics_names, res))

    loss = res['loss']
    error5_val = 1 - res[TOP_5]
    error1_val = 1 - res[TOP_1]

    return loss, error5_val, error1_val


def mst_eval_fn(epoch):
    def mst_eval_fn_helper(mst_state):
        """

        :param mst_state:

        Args:
            mst_state (TYPE): Description

        Returns:
            TYPE: Description
        """
        stop_list = []
        for mst_id in mst_state:
            if len(mst_state[mst_id]['train_loss']) == epoch:
                stop_list.append(mst_id)

        new_msts = []
        update_mst_id_list = []
        return stop_list, update_mst_id_list, new_msts
    return mst_eval_fn_helper
