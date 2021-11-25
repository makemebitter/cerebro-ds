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
import dill
from cerebro_gpdb.utils import cats
from cerebro_gpdb.utils import logs
from cerebro_gpdb.utils import set_seed
from cerebro_gpdb.in_rdbms_helper import main_prepare
from cerebro_gpdb.in_rdbms_helper import mst_2_str
from cerebro_gpdb.ctq import ConcurrentTargetedQueryClientBase
from cerebro_gpdb.ctq import ConcurrentTargetedQueryClient
from cerebro_gpdb.ctq import get_summary
from cerebro_gpdb.imagenetcat import param_grid_hyperopt as param_grid
from cerebro_gpdb.imagenetcat import SEED
from cerebro_gpdb.imagenetcat import MODEL_ARCH_TABLE
from cerebro_gpdb.hyperopt_helper import hyperopt_add_one_batch_configs
from exps.data_analytics import ctq_find
from exps.data_analytics import ctq_parse_model_info_ordered
import random
import numpy as np
from hyperopt.base import Domain
from hyperopt import hp
from hyperopt import Trials
from hyperopt import STATUS_OK


random.seed(SEED)
# Client does not use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ConcurrentTargetedQueryHyperopt(ConcurrentTargetedQueryClientBase):
    def __init__(self,
                 db_creds,
                 schema_madlib,
                 param_grid_hyperopt,
                 source_table,
                 validation_table,
                 model_arch_table,
                 use_gpus,
                 epochs,
                 models_root,
                 logs_root=None,
                 shuffle=True,
                 max_num_config=32,
                 concurrency=8):
        super(ConcurrentTargetedQueryHyperopt, self).__init__(
            db_creds,
            schema_madlib,
            None,
            source_table,
            validation_table,
            model_arch_table,
            use_gpus,
            epochs,
            models_root,
            logs_root=logs_root,
            shuffle=shuffle
        )

        self.param_grid_hyperopt = param_grid_hyperopt
        self.max_num_config = max_num_config
        self.concurrency = concurrency
        self.search_space = {
            'model':
            hp.choice('model', param_grid_hyperopt['model']),
            'lambda_value':
            hp.choice('lambda_value', param_grid_hyperopt['lambda_value']),
            'learning_rate':
            hp.loguniform('learning_rate',
                          np.log(param_grid_hyperopt['learning_rate'][0]),
                          np.log(param_grid_hyperopt['learning_rate'][1])),
            'batch_size':
            hp.choice(
                'batch_size',
                list(
                    range(param_grid_hyperopt['batch_size'][0],
                          param_grid_hyperopt['batch_size'][1] + 1)))
        }
        self.trials = Trials()
        self.domain = Domain(None, self.search_space)
        self.rand = np.random.RandomState(SEED)
        self.hyperopt_params = []
        self.msts = []

    def add_one_batch_configs(self):
        self.hyperopt_params, \
            self.msts, \
            self.new_start_idx, \
            self.new_end_idx = \
            hyperopt_add_one_batch_configs(
                self.param_grid_hyperopt, self.hyperopt_params, self.msts,
                self.domain, self.trials, self.concurrency)

    def get_curr_batch_msts(self):
        return self.msts[self.new_start_idx:self.new_end_idx]

    def update_hyperopt_curr_batch(self, model_info_ordered):
        df_grand = ctq_parse_model_info_ordered(model_info_ordered)
        for i, mst in enumerate(self.get_curr_batch_msts()):
            mst_key = mst_2_str(mst)
            self.hyperopt_params[i]['status'] = STATUS_OK
            _, y, _ = ctq_find(df_grand, mst_key=mst_key, best=False, mode='loss')
            self.hyperopt_params[i]['result'] = {
                'loss': y[-1],
                'status': STATUS_OK
            }
            self.trials.refresh()

    def run(self):
        # pseduo run training for epochs
        i = 0
        model_info_ordered_batch = {}
        return_dict_grand_batch = {}
        model_info_filepath = os.path.join(self.logs_root,
                                           "models_info_grand.pkl")
        jobs_info_filepath = os.path.join(self.logs_root,
                                          "jobs_info_grand.pkl")
        while len(self.hyperopt_params) < self.max_num_config:
            logs("STARTING BATCH:{}, FINISHED:{}".format(
                i, len(self.hyperopt_params)))
            self.add_one_batch_configs()
            ctq_client = ConcurrentTargetedQueryClient(
                db_creds=self.db_creds,
                schema_madlib=self.schema_madlib,
                msts=self.get_curr_batch_msts(),
                source_table=self.source_table,
                validation_table=self.validation_table,
                model_arch_table=self.model_arch_table,
                use_gpus=self.use_gpus,
                epochs=self.epochs,
                logs_root=None,
                models_root=self.models_root)
            model_info_ordered, return_dict_grand = ctq_client.run()
            model_info_ordered_batch[i] = model_info_ordered
            return_dict_grand_batch[i] = return_dict_grand
            self.update_hyperopt_curr_batch(model_info_ordered)
            summary = get_summary(ctq_client.model_info_ordered)
            print(summary)
            with open(model_info_filepath, "wb") as f:
                dill.dump(model_info_ordered_batch, f)
            with open(jobs_info_filepath, "wb") as f:
                dill.dump(return_dict_grand_batch, f)
            logs("ENDING BATCH:{}, FINISHED:{}".format(i,
                                                       len(
                                                           self.hyperopt_params
                                                       )))
            i += 1


if __name__ == '__main__':
    args, msts = main_prepare(shuffle=False)
    if args.run:
        logs("START RUNNING CTQ-HYPEROPT")
        set_seed(SEED)
        # args.train_name = args.valid_name
        print(args.train_name, args.valid_name)
        ctq_hyperopt_client = ConcurrentTargetedQueryHyperopt(
            db_creds=cats,
            schema_madlib='madlib',
            param_grid_hyperopt=param_grid,
            source_table=args.train_name,
            validation_table=args.valid_name,
            model_arch_table=MODEL_ARCH_TABLE,
            use_gpus=not args.criteo,
            epochs=args.num_epochs,
            logs_root=args.logs_root,
            models_root=args.models_root,
            shuffle=False,
            max_num_config=args.max_num_config,
            concurrency=args.size
        )
        ctq_hyperopt_client.run()
        logs("END RUNNING")
