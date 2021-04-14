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
from in_rdbms_helper import params_fac
from in_rdbms_helper import main_prepare
from in_rdbms_helper import mst_2_str
from utils import tstamp
import datetime
import time
from utils import DBBase
from utils import wait
from utils import cats
from imagenetcat import MODEL_SELECTION_TABLE
from imagenetcat import MODEL_SELECTION_SUMMARY_TABLE
from imagenetcat import MODEL_ARCH_TABLE


class ImageNetMOPRunner(object):
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
        print("START MOP: {}".format(tstamp()))
        mstt = MODEL_SELECTION_TABLE
        output = 't_{timestamp}_MOP'.format(**locals())
        output_summary = output + '_summary'
        output_model_info = output + '_model_info'
        
        query = '''
            DROP TABLE IF EXISTS {output},{output_summary},{output_model_info};

            SELECT madlib.madlib_keras_fit_multiple_model(
                '{self.train_name}',    -- source_table
                '{output}',     -- model_output_table
                '{mstt}',               -- model_selection_table
                {self.num_epochs},                       -- num_iterations
                {self.use_gpus},                     -- use gpus
                '{self.valid_name}',      -- validation dataset
                1                         -- metrics compute frequency
            );
        '''.format(**locals())
        self.acursor.execute(query)
        wait(self.aconnection)
        total_dur = time.time() - start
        print("END MOP: {}".format(tstamp()))
        print("END MOP DUR: {}".format(total_dur))

    def load_models(self):
        print("PURGING MODEL SELECTION TABLE")
        mstt = MODEL_SELECTION_TABLE
        mstst = MODEL_SELECTION_SUMMARY_TABLE
        self.cursor.execute(
            """
            DROP TABLE IF EXISTS {mstt};
            DROP TABLE IF EXISTS {mstst};
            DROP TABLE IF EXISTS {mstt}, {mstst};
            SELECT madlib.load_model_selection_table(
                'model_arch_library', -- model architecture table
                '{mstt}',          -- model selection table output
                ARRAY[1],          -- model ids from model architecture table
                ARRAY[               -- compile params   
                    $$ loss='categorical_crossentropy',optimizer='rmsprop(lr=0.0001, decay=1e-6)',metrics=['accuracy'] $$
                ],
                ARRAY[                -- fit params
                    $$ batch_size=64,epochs=5 $$
                ]
            );
            TRUNCATE TABLE {mstt};
            alter table {mstt} DROP CONSTRAINT mst_table_model_id_key;
            """.format(**locals())
        )
        wait(self.connection)
        for i, mst in enumerate(self.msts):
            i += 1
            self.load_model(i, mst)

    def load_model(self, i, mst):
        compile_params, fit_params = params_fac(mst)
        self.cursor.execute(
            "SELECT model_id from {} WHERE name=$${}$$".format(
                MODEL_ARCH_TABLE, mst_2_str(mst)))
        model_id = self.cursor.fetchone()[0]
        mstt = MODEL_SELECTION_TABLE
        query = """INSERT INTO {mstt}(mst_key, model_id, compile_params, fit_params)
            VALUES
                ({i}, {model_id}, {compile_params}, {fit_params})""".format(
            **locals())
        self.cursor.execute(query)
        wait(self.connection)


if __name__ == "__main__":
    args, msts = main_prepare(shuffle=True)
    print(args.train_name, args.valid_name)
    runner = ImageNetMOPRunner(cats, msts, args.num_epochs,
                               args.train_name, args.valid_name, no_gpu=args.criteo)
    if args.load:
        print("LOADING MODELS")
        runner.load_models()
    if args.run:
        print("RUNNING EXPS")
        runner.run()
