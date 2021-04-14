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
from in_rdbms_helper import main_prepare
from in_rdbms_helper import params_fac_hyperopt
from utils import tstamp
import datetime
import time
from utils import wait
from utils import cats
from utils import get_output_names_hyperopt
from imagenetcat import MODEL_SELECTION_TABLE
from imagenetcat import MODEL_ARCH_TABLE
from imagenetcat import param_grid_hyperopt
from db_runner import DBRunner


class UDAFHyperoptRunner(DBRunner):
    def run(self):
        start = time.time()
        timestamp = datetime.datetime.fromtimestamp(
            start).strftime('%Y_%m_%d_%H_%M_%S')
        print("START UDAF-Hyperopt: {}".format(tstamp()))
        mstt = MODEL_SELECTION_TABLE
        output, output_info, output_summary, \
            output_mst_table, output_mst_table_summary = \
            get_output_names_hyperopt(timestamp)
        mat = MODEL_ARCH_TABLE
        model_ids = "ARRAY{}".format(str(list(range(1, len(self.msts) + 1))))
        self.num_configs = 32
        compile_param_grid, fit_params_grid = \
            params_fac_hyperopt(param_grid_hyperopt)
        query = '''
            DROP TABLE IF EXISTS {output},{output_info},{output_summary},{output_mst_table},{output_mst_table_summary};
            SELECT madlib.madlib_keras_automl('{self.train_name}',                -- source table
                                  '{output}',                    -- model output table
                                  '{mat}',               -- model architecture table
                                  '{output_mst_table}',                 -- model selection output table
                                  {model_ids},                         -- model IDs
                                  {compile_param_grid},                               -- compile param grid
                                  {fit_params_grid},  -- fit params grid
                                  'hyperopt',                         -- autoML method
                                  'num_configs={self.num_configs}, num_iterations={self.num_epochs}, algorithm=tpe',  -- autoML params
                                  {self.seed},                               -- random state
                                  NULL,                               -- object table
                                  {self.use_gpus},                              -- use GPUs
                                  '{self.valid_name}',                 -- validation table
                                  1,                                  -- metrics compute freq
                                  NULL,                               -- name
                                  NULL                                -- description  
                                  );
        '''.format(**locals())
        self.acursor.execute(query)
        wait(self.aconnection)
        total_dur = time.time() - start
        print("END UDAF-Hyperopt: {}".format(tstamp()))
        print("END UDAF-Hyperopt DUR: {}".format(total_dur))


if __name__ == "__main__":
    args, msts = main_prepare(shuffle=False)
    print(args.train_name, args.valid_name)
    runner = UDAFHyperoptRunner(
        cats, msts, args.num_epochs,
        args.train_name, args.valid_name, no_gpu=args.criteo)
    if args.load:
        print("LOADING MODELS")
        runner.load_models(msts)
    if args.run:
        print("RUNNING EXPS")
        runner.run()
