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
from cerebro_gpdb.imagenetcat import TOP_5
from cerebro_gpdb.imagenetcat import TOP_1
from cerebro_gpdb.imagenetcat import INPUT_SHAPE
from cerebro_gpdb.imagenetcat import NUM_CLASSES
from cerebro_gpdb.imagenetcat import SEED
from cerebro_gpdb.imagenetcat import param_grid_hyperopt
from cerebro_gpdb.criteocat import INPUT_SHAPE as INPUT_SHAPE_CRITEO
from cerebro_gpdb.criteocat import NUM_CLASSES as NUM_CLASSES_CRITEO
from cerebro_gpdb.in_rdbms_helper import create_model_from_mst
from hyperopt import tpe, STATUS_OK, STATUS_RUNNING
import h5py
import numpy as np
import argparse
from multiprocessing.dummy import Pool

DATASET_NAME = {
    'imagenet': 'imagenet_cerebro_standalone',
    'imagenet_4': 'imagenet_cerebro_standalone_4',
    'imagenet_2': 'imagenet_cerebro_standalone_2',
    'imagenet_1': 'imagenet_cerebro_standalone_1',
    'criteo': 'criteo_cerebro_standalone'
}
DATA_CATALOG = {
    DATASET_NAME['imagenet']: {
        'data_root': '/mnt/imagenet',
        'train': ['train/train_0.h5', 'train/train_1.h5', 'train/train_2.h5', 'train/train_3.h5',
                  'train/train_4.h5', 'train/train_5.h5', 'train/train_6.h5', 'train/train_7.h5'
                  ],
        'train_availability': [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        'valid': ['valid/valid_0.h5', 'valid/valid_1.h5', 'valid/valid_2.h5', 'valid/valid_3.h5',
                  'valid/valid_4.h5', 'valid/valid_5.h5', 'valid/valid_6.h5', 'valid/valid_7.h5'
                  ],
        'valid_availability': [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    },
    DATASET_NAME['imagenet_4']: {
        'data_root': '/mnt/imagenet',
        'train': ['train/train_0.h5', 'train/train_1.h5', 'train/train_2.h5', 'train/train_3.h5',
                  'train/train_4.h5', 'train/train_5.h5', 'train/train_6.h5', 'train/train_7.h5'
                  ],
        'train_availability': [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        'valid': ['valid/valid_0.h5', 'valid/valid_1.h5', 'valid/valid_2.h5', 'valid/valid_3.h5',
                  'valid/valid_4.h5', 'valid/valid_5.h5', 'valid/valid_6.h5', 'valid/valid_7.h5'
                  ],
        'valid_availability': [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    },
    DATASET_NAME['imagenet_2']: {
        'data_root': '/mnt/imagenet',
        'train': ['train/train_0.h5', 'train/train_1.h5', 'train/train_2.h5', 'train/train_3.h5',
                  'train/train_4.h5', 'train/train_5.h5', 'train/train_6.h5', 'train/train_7.h5'
                  ],
        'train_availability': [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        'valid': ['valid/valid_0.h5', 'valid/valid_1.h5', 'valid/valid_2.h5', 'valid/valid_3.h5',
                  'valid/valid_4.h5', 'valid/valid_5.h5', 'valid/valid_6.h5', 'valid/valid_7.h5'
                  ],
        'valid_availability': [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    },
    DATASET_NAME['imagenet_1']: {
        'data_root': '/mnt/imagenet',
        'train': ['train/train_0.h5', 'train/train_1.h5', 'train/train_2.h5', 'train/train_3.h5',
                  'train/train_4.h5', 'train/train_5.h5', 'train/train_6.h5', 'train/train_7.h5'
                  ],
        'train_availability': [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        'valid': ['valid/valid_0.h5', 'valid/valid_1.h5', 'valid/valid_2.h5', 'valid/valid_3.h5',
                  'valid/valid_4.h5', 'valid/valid_5.h5', 'valid/valid_6.h5', 'valid/valid_7.h5'
                  ],
        'valid_availability': [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    },
    DATASET_NAME['criteo']: {
        'data_root': '/mnt/hdd/criteo',
        'train': ['train/train_0.npy', 'train/train_1.npy', 'train/train_2.npy', 'train/train_3.npy',
                  'train/train_4.npy', 'train/train_5.npy', 'train/train_6.npy', 'train/train_7.npy'
                  ],
        'train_availability': [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        'valid': ['valid/valid_0.npy', 'valid/valid_1.npy', 'valid/valid_2.npy', 'valid/valid_3.npy',
                  'valid/valid_4.npy', 'valid/valid_5.npy', 'valid/valid_6.npy', 'valid/valid_7.npy'
                  ],
        'valid_availability': [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    }
}

# def get_main_parser():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("-l", "--logs_root", nargs='?', default="./logs",
#                         help="Directory for storing log files")
#     parser.add_argument("-e", "--num_epochs", nargs='?', default=10, type=int,
#                         help="Number of training epochs to perform")
#     parser.add_argument(
#         '--train_name', type=str, default='imagenet_train_data_packed'
#     )
#     parser.add_argument(
#         '--valid_name', type=str, default='imagenet_valid_data_packed'
#     )
#     parser.add_argument(
#         '--db_name', type=str, default='cerebro'
#     )
#     parser.add_argument(
#         '--run', action='store_true'
#     )
#     parser.add_argument(
#         '--best_model_run', action='store_true'
#     )
#     parser.add_argument(
#         '--models_root', type=str
#     )
#     return parser


def get_dataset(data, mst, generator_data):
    import tensorflow as tf
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


def mst_eval_fn_hyperopt(
    epoch,
    hyperopt_params,
    trials,
    domain,
    max_num_config,
    rand,
    concurrency,
    model_options,
    param_grid=param_grid_hyperopt
):
    def mst_eval_fn_helper(mst_state):
        """

        :param mst_state:

        Args:
            mst_state (TYPE): Description

        Returns:
            TYPE: Description
        """
        stop_list = []
        generate_new = False
        for mst_id in mst_state:
            if len(mst_state[mst_id]['train_loss']) == epoch \
                and mst_state[mst_id][
                    "state"] == "RUNNING":
                stop_list.append(mst_id)

                hyperopt_params[mst_id]['status'] = STATUS_OK
                hyperopt_params[mst_id]['result'] = {
                    'loss': mst_state[mst_id]['valid_loss'][-1],
                    'status': STATUS_OK}
                trials.refresh()

                generate_new = True

        new_msts = []
        if generate_new and len(mst_state) < max_num_config:
            for j in range(len(mst_state), len(mst_state)+concurrency):
                new_param = tpe.suggest(
                    [j], domain, trials, rand.randint(0, 2 ** 31 - 1))
                new_param[0]['status'] = STATUS_RUNNING

                trials.insert_trial_docs(new_param)
                trials.refresh()
                hyperopt_params.append(new_param[0])

            # Generating Cerebro params from HyperOpt params
            new_msts = []
            for hyperopt_param in hyperopt_params[-concurrency:]:
                param = {}
                for k in hyperopt_param['misc']['vals']:
                    val = hyperopt_param['misc']['vals'][k][0].item()
                    if k in ["model", 'lambda_value']:
                        # if the hyperparamer is a choice the index is returned
                        val = param_grid[k][val]
                    param[k] = val
                new_msts.append(param)
        return stop_list, [], new_msts
    return mst_eval_fn_helper


def input_fn(file_path, one_hot=True):
    """

    :param file_path:
    :return:
    """
    h5f = h5py.File(file_path, 'r')

    images = h5f.get("images")

    size = images.shape[0]

    n_workers = 32
    partition_size = int(np.ceil(size / float(n_workers)))
    h5f.close()

    def _read_data(x):
        file_path, start, partition_size = x
        h5f = h5py.File(file_path, 'r')
        np_images = h5f.get("images")[start:start+partition_size]
        if one_hot:
            np_labels = np.eye(1000)[h5f.get("labels")[
                start:start+partition_size].astype(int)]
        else:
            np_labels = np.asarray(h5f.get("labels")[
                start:start+partition_size].astype(int))
        h5f.close()

        return {"images": np_images, "labels": np_labels}

    pool = Pool(n_workers)
    result = pool.map(
        _read_data, ((file_path, x, partition_size)
                     for x in list(range(0, size, partition_size))))
    pool.close()
    pool.join()

    return result


def input_fn_criteo(file_path):
    """

    :param file_path:
    :return:
    """
    np_arr = np.load(file_path)
    np_images = np_arr[:, :-1].astype(np.float32)
    np_labels = np_arr[:, -1].astype(int)
    np_labels = np.eye(2)[np_labels]
    return {"images": np_images, "labels": np_labels}


class generator_data:
    def __init__(self, data):
        self.data = data
        self.length = self.get_df_length(self.data)

    def get_df_length(self, data):
        length = 0
        for x in data:
            length += x["images"].shape[0]
        return length

    def __call__(self):
        for x in self.data:
            for i in range(x["images"].shape[0]):
                image_arr = x["images"][i]
                label = x["labels"][i]
                yield image_arr, label


class generator_data_criteo:
    def __init__(self, data):
        self.data = data
        self.length = self.data['labels'].shape[0]

    def __call__(self):
        for i in range(self.data["images"].shape[0]):
            image_arr = self.data["images"][i]
            label = self.data["labels"][i]
            yield image_arr, label
