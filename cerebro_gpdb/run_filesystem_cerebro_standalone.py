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
from cerebro_gpdb.utils import logs
from cerebro_gpdb.run_cerebro_standalone_helper import model_fn
from cerebro_gpdb.run_cerebro_standalone_helper import train_fn
from cerebro_gpdb.run_cerebro_standalone_helper import mst_eval_fn as mst_eval_fn_grid
from cerebro_gpdb.run_cerebro_standalone_helper import mst_eval_fn_hyperopt
from cerebro_gpdb.run_cerebro_standalone_helper import DATASET_NAME
from cerebro_gpdb.run_cerebro_standalone_helper import DATA_CATALOG
from cerebro_gpdb.run_cerebro_standalone_helper import generator_data_criteo
from cerebro_gpdb.run_cerebro_standalone_helper import input_fn_criteo
from cerebro_gpdb.run_cerebro_standalone_helper import generator_data
from cerebro_gpdb.run_cerebro_standalone_helper import input_fn
from cerebro_gpdb.hyperopt_helper import init_hyperopt
from cerebro_gpdb.in_rdbms_helper import main_prepare
from cerebro_gpdb.imagenetcat import param_grid_hyperopt
import os
import numpy as np


def main(args, msts):
    logs(msts)
    print("HELLO")
    if args.criteo:
        dataset_name = DATASET_NAME['criteo']
        generator_data_cls = generator_data_criteo
        input_fn_pass = input_fn_criteo
    else:
        if args.size != 8:
            dtname = 'imagenet_{}'.format(args.size)
        else:
            dtname = 'imagenet'
        dataset_name = DATASET_NAME[dtname]
        generator_data_cls = generator_data
        input_fn_pass = input_fn
    data_catalog = DATA_CATALOG
    if args.sanity:
        data_catalog[dataset_name]['train'] = data_catalog[dataset_name]['valid']
    if args.run:
        logs("START RUNNING")
        if not os.path.exists(args.models_root):
            logs("MAKING models_root")
            os.makedirs(args.models_root)
        if not os.path.exists(args.logs_root):
            logs("MAKING logs_root")
            os.makedirs(args.logs_root)
        if args.cerebro_spark:
            if args.hyperopt:
                from cerebro_gpdb.cerebro_spark_wrapper import schedule_hyperopt
                schedule_hyperopt(args)
            else:
                from cerebro_gpdb.cerebro_spark_wrapper import schedule_grid_search
                schedule_grid_search(args)
        else:
            from cerebro.code.client import schedule
            if args.hyperopt:
                hyperopt_params, msts, trials, domain, rand, model_options = \
                    init_hyperopt(param_grid_hyperopt, args)
                mst_eval_fn = mst_eval_fn_hyperopt(
                    args.num_epochs,
                    hyperopt_params,
                    trials,
                    domain,
                    args.max_num_config,
                    rand,
                    args.size,
                    model_options)
            else:
                mst_eval_fn = mst_eval_fn_grid(args.num_epochs)
            schedule(
                data_set_name=dataset_name,
                input_fn=input_fn_pass,
                model_fn=model_fn(generator_data_cls),
                train_fn=train_fn,
                initial_msts=msts,
                mst_eval_fn=mst_eval_fn,
                ckpt_root=args.models_root,
                preload_data_to_mem=True,
                log_files_root=args.logs_root,
                backend='keras',
                data_catalog=data_catalog,
                save_every=args.best_model_run
            )

        logs("END RUNNING")


if __name__ == '__main__':
    args, msts = main_prepare(shuffle=False)
    main(args, msts)
