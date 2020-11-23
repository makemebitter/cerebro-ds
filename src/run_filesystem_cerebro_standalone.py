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

from __future__ import division
from __future__ import print_function
from cerebro_gpdb.pathmagic import *  # noqa
from cerebro_gpdb.in_rdbms_helper import get_exp_specific_msts
from cerebro_gpdb.utils import logs
from cerebro_gpdb.run_cerebro_standalone_helper import model_fn
from cerebro_gpdb.run_cerebro_standalone_helper import train_fn
from cerebro_gpdb.run_cerebro_standalone_helper import mst_eval_fn
from cerebro_gpdb.in_rdbms_helper import main_prepare
from cerebro.code.client import schedule
import os
import h5py
import numpy as np
from multiprocessing.dummy import Pool

DATASET_NAME = {
    'imagenet':'imagenet_cerebro_standalone',
    'criteo':'criteo_cerebro_standalone'
    }
DATA_CATALOG = {
    DATASET_NAME['imagenet']: {
        'data_root': '/mnt/imagenet',
        'train': ['train_0.h5', 'train_1.h5', 'train_2.h5', 'train_3.h5',
                  'train_4.h5', 'train_5.h5', 'train_6.h5', 'train_7.h5'
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
        'valid': ['valid_0.h5', 'valid_1.h5', 'valid_2.h5', 'valid_3.h5',
                  'valid_4.h5', 'valid_5.h5', 'valid_6.h5', 'valid_7.h5'
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
    DATASET_NAME['criteo']: {
        'data_root': '/mnt/criteo',
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


def input_fn(file_path):
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
        np_labels = np.eye(1000)[h5f.get("labels")[
            start:start+partition_size].astype(int)]
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

if __name__ == '__main__':
    args, msts = main_prepare(shuffle=False)
    logs(msts)
    print("HELLO")
    if args.criteo:
        dataset_name = DATASET_NAME['criteo']
        generator_data_cls = generator_data_criteo
        input_fn_pass = input_fn_criteo
    else:
        dataset_name = DATASET_NAME['imagenet']
        generator_data_cls = generator_data
        input_fn_pass = input_fn
    data_catalog = DATA_CATALOG
    if args.run:
        logs("START RUNNING")
        if not os.path.exists(args.models_root):
            logs("MAKING models_root")
            os.makedirs(args.models_root)
        if not os.path.exists(args.logs_root):
            logs("MAKING logs_root")
            os.makedirs(args.logs_root)

        schedule(
            data_set_name=dataset_name,
            input_fn=input_fn_pass,
            model_fn=model_fn(generator_data_cls),
            train_fn=train_fn,
            initial_msts=msts,
            mst_eval_fn=mst_eval_fn(args.num_epochs),
            ckpt_root=args.models_root,
            preload_data_to_mem=True,
            log_files_root=args.logs_root,
            backend='keras',
            data_catalog=data_catalog,
            save_every=args.best_model_run
        )

        logs("END RUNNING")
