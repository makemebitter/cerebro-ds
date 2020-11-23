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

import datetime
import time
from utils import DBBase
from utils import wait
from utils import cats
from utils import tstamp
from in_rdbms_helper import mst_2_str
from in_rdbms_helper import params_fac
from in_rdbms_helper import main_prepare
from in_rdbms_helper import create_model_from_mst

from imagenetcat import MODEL_ARCH_TABLE


class ImageNetRunner(object):
    def __init__(self, db_creds, msts, num_epochs, train_name, valid_name, no_gpu=False):
        self.adb = DBBase(db_creds, 1)
        self.aconnection = self.adb.connection
        self.acursor = self.adb.cursor
        self.db = DBBase(db_creds, 0)
        self.connection = self.db.connection
        self.cursor = self.db.cursor
        self.msts = msts
        self.num_epochs = num_epochs
        self.train_name = train_name
        self.valid_name = valid_name
        self.no_gpu = no_gpu
        self.use_gpus = 'FALSE' if self.no_gpu else 'TRUE'

    def run(self):
        start = time.time()
        timestamp = datetime.datetime.fromtimestamp(
            start).strftime('%Y_%m_%d_%H_%M_%S')
        print("START MODEL AVERAGING: {}".format(tstamp()))
        for mst in self.msts:
            self.run_one(mst, timestamp)
        total_dur = time.time() - start
        print("END MODEL AVERAGING: {}".format(tstamp()))
        print("END MODEL AVERAGING DUR: {}".format(total_dur))

    def load_models(self, purge=True):
        if purge:
            print("PURGING MODEL ARCH TABLE")
            self.cursor.execute(
                """DROP TABLE IF EXISTS {};""".format(MODEL_ARCH_TABLE))
            wait(self.connection)
        for mst in msts:
            self.load_model(mst)

    def load_model(self, mst):
        model = create_model_from_mst(mst)
        query = "SELECT madlib.load_keras_model('{}', %s, NULL, %s)".format(
            MODEL_ARCH_TABLE)
        self.cursor.execute(query, [model.to_json(), mst_2_str(mst)])
        wait(self.connection)

    def run_one(self, mst, timestamp):
        mst_str = mst_2_str(mst)
        mst_start = time.time()

        model_arch_library = MODEL_ARCH_TABLE

        print("START MODEL AVERAGING MODEL-{} : {}".format(mst_str, tstamp()))
        self.cursor.execute(
            """SELECT model_id from {} WHERE name = '{}'""".format(
                MODEL_ARCH_TABLE, mst_str))
        model_id = self.cursor.fetchone()[0]
        model_table = "T_{}_M_{}".format(timestamp, model_id)
        model_summary_table = model_table + '_summary'
        self.cursor.execute(
            """DROP TABLE IF EXISTS {model_table}, {model_summary_table};""".
            format(**locals()))

        compile_params, fit_params = params_fac(mst)
        wait(self.connection)
        self.acursor.execute("""
            SELECT madlib.madlib_keras_fit(
                '{self.train_name}',    -- source table
                '{model_table}',                -- model output table
                '{model_arch_library}',            -- model arch table
                {model_id},                              -- model arch id
                {compile_params},  -- compile_params
                {fit_params},  -- fit_params
                {self.num_epochs},                    -- num_iterations
                {self.use_gpus},                          -- use GPUs
                '{self.valid_name}',    -- validation dataset
                1                               -- metrics compute frequency
            );""".format(**locals()))
        wait(self.aconnection)
        print("END MODEL AVERAGING MODEL-{} : {}".format(mst_str, tstamp()))
        mst_dur = time.time() - mst_start
        print("END MODEL AVERAGING MODEL-{} DUR: {}".format(mst_str, mst_dur))


if __name__ == "__main__":
    args, msts = main_prepare(shuffle=False)
    runner = ImageNetRunner(cats, msts, args.num_epochs,
                            args.train_name, args.valid_name, args.criteo)
    if args.load:
        print("LOADING MODELS")
        runner.load_models()
    if args.run:
        print("RUNNING EXPS")
        runner.run()
