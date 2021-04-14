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
import subprocess
from cerebro_gpdb.imagenetcat import param_grid_hyperopt
from cerebro_gpdb.hyperopt_helper import init_hyperopt
from cerebro_gpdb.in_rdbms_helper import main_prepare
from cerebro_gpdb.utils import logs as plogs
from pyspark.sql import SparkSession
import os
from hyperopt import SparkTrials
from hyperopt import fmin, tpe
import sys
sys.path.append('/local')
sys.path.append('/local/cerebro-greenplum')
# import tensorflow as tf
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3.7'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3.7'
os.environ['PYTHONPATH'] = '/local:/local/cerebro-greenplum'


def train_fn_fac(train_list, valid_list, epochs=10, logs_dir=''):
    def train_fn(mst):
        try:
            from hyperopt import STATUS_OK
            import sys
            import os
            import traceback
            import tensorflow.keras as keras
            sys.path.append('/local')
            sys.path.append('/local/cerebro-greenplum')
            from cerebro_gpdb.utils import DiskLogs
            from cerebro_gpdb.utils import mst2key
            from cerebro_gpdb.utils import logsc
            from cerebro_gpdb.utils import LOG_KEYS
            from cerebro_gpdb.single_node_helper import data_h5
            from cerebro_gpdb.single_node_helper import compile_model_from_mst
            from cerebro_gpdb.single_node_helper import RefreshOptimizer
            from cerebro_gpdb.in_rdbms_helper import create_model_from_mst
            from cerebro_gpdb.imagenetcat import SEED
            from cerebro_gpdb.in_rdbms_helper import set_seed
            set_seed(SEED, 'tf.keras')
            mst_key = mst2key(mst)
            log_filenames = [os.path.join(logs_dir, mst_key + '.log'),
                             os.path.join(
                                 logs_dir, os.environ['WORKER_NAME'] + '.log')
                             ]
            logs = DiskLogs(log_filenames)
            logs("MST: {}".format(mst_key))
            with logsc(LOG_KEYS.DATA_LOADING, logs_fn=logs):
                train_dataset, valid_dataset, train_steps, valid_steps = \
                    data_h5(
                        epochs, mst, train_list, valid_list)
            with logsc(LOG_KEYS.MODEL_INIT, logs_fn=logs):
                model = create_model_from_mst(mst, module='tf.keras')
                model = compile_model_from_mst(mst, model)

            # train_steps = valid_steps = 10
            with logsc(LOG_KEYS.MODEL_TRAINVALID, logs_fn=logs):
                history = model.fit(x=train_dataset,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=valid_dataset,
                                    steps_per_epoch=train_steps,
                                    validation_steps=valid_steps,
                                    callbacks=[RefreshOptimizer(mst)]
                                    )
            # callbacks=[]
            # RefreshOptimizer(mst)
            logs("HISTORY: {}".format(history.history))
            return {
                'loss': history.history['val_loss'][-1], 'status': STATUS_OK}
        except Exception as e:
            print("WRONG")
            traceback.print_exc()
            raise e
    return train_fn


if __name__ == '__main__':
    args, _ = main_prepare(shuffle=False)
    spark = SparkSession \
        .builder \
        .master("spark://10.10.1.1:7077") \
        .config('spark.python.worker.memory', "150G")\
        .config("spark.executor.memory", "150G")\
        .config("spark.executorEnv.HADOOP_HOME", "/local/hadoop")\
        .config('spark.default.parallelism', str(args.size))\
        .config("spark.executor.cores", "1")\
        .config('spark.executor.pyspark.memory', '150G')\
        .appName("ImageNet") \
        .getOrCreate()
    search_space, rand = init_hyperopt(
        param_grid_hyperopt, args, search_space_only=True)
    spark_trials = SparkTrials(spark_session=spark, parallelism=args.size)
    app_id = spark.sparkContext.applicationId
    plogs("APP ID: {}".format(app_id))
    print("TAIL Commands: tail -f /usr/local/spark/work/{}/?/stderr".format(
        app_id))
    print("TAIL Commands: tail -f {}/$WORKER_NAME.log".format(
        args.logs_root))
    best_hyperparameters = fmin(
        fn=train_fn_fac(None, None, args.num_epochs, args.logs_root),
        space=search_space,
        algo=tpe.suggest,
        trials=spark_trials,
        max_evals=args.max_num_config,
        rstate=rand
    )
    plogs(best_hyperparameters)
    # plogs("COLLECTING LOGS ...")
    # cmd = "bash /local/cerebro-greenplum/cerebro_gpdb/collect_spark_logs.sh {} {}".format(
    #     app_id, args.logs_root)
    # process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
