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
from cerebro_gpdb.pathmagic import * # noqa
from cerebro_gpdb.utils import cats
from cerebro_gpdb.da import DirectAccessClient
from cerebro_gpdb.da import input_fn
from cerebro_gpdb.run_cerebro_standalone_helper import model_fn
from cerebro_gpdb.run_cerebro_standalone_helper import train_fn
from cerebro_gpdb.run_cerebro_standalone_helper import mst_eval_fn
from cerebro_gpdb.utils import logs
from in_rdbms_helper import main_prepare
from cerebro.code.client import schedule
import os
import dill

class generator_df:
    def __init__(self, df_actual_data):
        self.df_actual_data = df_actual_data
        self.length = self.get_df_length(self.df_actual_data)

    def get_df_length(self, df_actual_data):
        length = 0
        for buffer_id, buffer in self.df_actual_data.items():
            dependent_var = buffer['dependent_var']
            independent_var = buffer['independent_var']
            assert dependent_var.shape[0] == independent_var.shape[0]
            len_buffer = dependent_var.shape[0]
            length += len_buffer
        return length

    def __call__(self):
        for buffer_id, buffer in self.df_actual_data.items():
            dependent_var = buffer['dependent_var']
            independent_var = buffer['independent_var']
            len_buffer = dependent_var.shape[0]
            for i in range(len_buffer):
                image_arr = independent_var[i]
                label = dependent_var[i]
                yield image_arr, label


if __name__ == '__main__':
    args, msts = main_prepare(shuffle=False)
    print("HELLO")
    if args.run:
        logs("START RUNNING")
        if not os.path.exists(args.models_root):
            logs("MAKING models_root")
            os.makedirs(args.models_root)
        if not os.path.exists(args.logs_root):
            logs("MAKING logs_root")
            os.makedirs(args.logs_root)
        DATASET_NAME = 'criteo_da' if args.criteo else 'imagenet_da'
        print(args.train_name, args.valid_name)
        da = DirectAccessClient(
            cats, args.db_name, args.train_name, args.valid_name, args.size)
        data_cat, sys_cats = da.generate_cats()
        
        workers = da.get_workers()
        # for sanitys check
        # data_cat['train'] = data_cat['valid']

        print (data_cat)
        data_cat = {DATASET_NAME: data_cat}
        schedule(
            data_set_name=DATASET_NAME,
            input_fn=input_fn,
            model_fn=model_fn(generator_df),
            train_fn=train_fn,
            initial_msts=msts,
            mst_eval_fn=mst_eval_fn(args.num_epochs),
            ckpt_root=args.models_root,
            preload_data_to_mem=True,
            log_files_root=args.logs_root,
            backend='keras',
            data_catalog=data_cat,
            workers=workers
        )

        logs("END RUNNING")
