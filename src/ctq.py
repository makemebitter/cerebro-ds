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

from utils import wait
from utils import cats
from utils import DBConnect
import time
import datetime
import os
import dill
from utils import CUDA_VISIBLE_DEVICES_KEY
from utils import get_initial_weights
from utils import logs
from in_rdbms_helper import main_prepare
from in_rdbms_helper import mst_2_str
from in_rdbms_helper import params_fac
from in_rdbms_helper import create_model_from_mst
from utils import set_seed
from imagenetcat import SEED
import traceback
import random
from multiprocessing import Process
from multiprocessing import Manager
from collections import defaultdict
import numpy as np
random.seed(SEED)
MODULE_NAME = 'ctq'
# Client does not use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_summary(model_info_ordered):
    summary = {}
    for k, v in model_info_ordered.items():
        summary[k] = defaultdict(list)
        for x in v:
            epoch = x['epoch']
            summary[k][epoch].append(x['metric_valid'])

    for k, v in summary.items():
        summary[k] = [np.mean(ll) for e, ll in v.items()]

    return summary


class ConcurrentTargetedQueryService(DBConnect):
    def __init__(self, db_creds, schema_madlib, dist_key, dist_key_mapping,
                 seg_ids_val, model_arch, compile_params, fit_params,
                 serialized_weights_dir, accessible_gpus_for_seg,
                 images_per_seg_train, images_per_seg_valid, source_table,
                 validation_table, use_gpus):
        super(ConcurrentTargetedQueryService, self).__init__(db_creds)
        self.schema_madlib = schema_madlib
        self.dist_key = dist_key
        self.dist_key_mapping = dist_key_mapping
        self.seg_ids_val = seg_ids_val
        self.model_arch = model_arch
        self.compile_params = compile_params
        self.fit_params = fit_params
        self.images_per_seg_train = images_per_seg_train
        self.images_per_seg_valid = images_per_seg_valid
        self.use_gpus = use_gpus
        self.accessible_gpus_for_seg = accessible_gpus_for_seg
        self.source_table = source_table
        self.validation_table = validation_table
        self.serialized_weights_dir = serialized_weights_dir

    def train_prepare(self):
        self.train_plan_name = "train_plan_{}".format(self.dist_key)
        prepare_statement_string = """
            PREPARE {self.train_plan_name} as
            SELECT {self.schema_madlib}.fit_step_ctq(
                {self.mb_dep_var_col},
                {self.mb_indep_var_col},
                {self.dep_shape_col},
                {self.ind_shape_col},
                $MAD${self.model_arch}$MAD$::TEXT,
                {self.compile_params}::TEXT,
                {self.fit_params}::TEXT,
                {self.dist_key_col},
                ARRAY{self.dist_key_mapping},
                gp_segment_id,
                {self.segments_per_host},
                ARRAY{self.images_per_seg_train},
                {self.use_gpus}::BOOLEAN,
                ARRAY{self.accessible_gpus_for_seg},
                $1::TEXT,
                $2::BOOLEAN
            ) AS iteration_result
            FROM {self.source_table}
            WHERE __dist_key__ = {self.dist_key}
            """.format(**locals())
        self.acursor.execute(prepare_statement_string)
        wait(self.aconnection)

    def train(self):

        #       Always run one sub-epoch and stop
        is_final_iteration = True
        execture_query_string = """execute {} (%s, %s)""".format(
            self.train_plan_name)
        self.acursor.execute(execture_query_string,
                             (self.serialized_weights_dir, is_final_iteration))
        wait(self.aconnection)
        self.serialized_weights = self.acursor.fetchone()[0]
        self.serialized_weights = bytes(self.serialized_weights)
        return self.serialized_weights

    def valid_prepare(self):
        self.valid_plan_t_name = "valid_plan_t_{}".format(self.dist_key)
        self.valid_plan_v_name = "valid_plan_v_{}".format(self.dist_key)
        for valid_plan_name, table_name, images_per_seg, dist_key_mapping \
            in zip(
                [self.valid_plan_t_name, self.valid_plan_v_name],
                [self.source_table, self.validation_table],
                [self.images_per_seg_train, self.images_per_seg_valid],
                [self.dist_key_mapping, self.seg_ids_val]):
            eval_prepare_string = """
                PREPARE {valid_plan_name} as
                select ({self.schema_madlib}.internal_keras_evaluate_ctq(
                                                    {self.mb_dep_var_col},
                                                    {self.mb_indep_var_col},
                                                    {self.dep_shape_col},
                                                    {self.ind_shape_col},
                                                    $MAD${self.model_arch}$MAD$,
                                                    $1::TEXT,
                                                    {self.compile_params},
                                                    {self.dist_key_col},
                                                    ARRAY{dist_key_mapping},
                                                    {self.gp_segment_id_col},
                                                    {self.segments_per_host},
                                                    ARRAY{images_per_seg},
                                                    {self.use_gpus}::BOOLEAN,
                                                    ARRAY{self.accessible_gpus_for_seg},
                                                    $2::BOOLEAN
                                                    )) as loss_metric
                from {table_name}
                WHERE __dist_key__ = {self.dist_key}
                """.format(**locals())
            self.acursor.execute(eval_prepare_string)
            wait(self.aconnection)

    def valid(self, on_train):
        is_final_iteration = True
        valid_plan_name = \
            self.valid_plan_t_name if on_train else self.valid_plan_v_name
        execture_query_string = """execute {} (%s, %s)""".format(
            valid_plan_name)
        self.acursor.execute(execture_query_string,
                             (self.serialized_weights_dir, is_final_iteration))
        wait(self.aconnection)

        evaluate_result = self.acursor.fetchone()[0]

        if len(evaluate_result) not in [1, 2]:
            raise Exception(
                'Calling evaluate on table returned < 2 '
                'metrics. Expected both loss and a metric.')
        loss = evaluate_result[0]
        metric = evaluate_result[1]

        return loss, metric


class ConcurrentTargetedQueryClient(DBConnect):
    def __init__(self, db_creds, schema_madlib, msts, source_table,
                 validation_table, model_arch_table,
                 use_gpus, epochs, models_root, logs_root=None):
        self.schema_madlib = schema_madlib
        super(ConcurrentTargetedQueryClient, self).__init__(db_creds)
        self.db_creds = db_creds
        self.msts = msts
        self.use_gpus = use_gpus
        self.epochs = epochs
        self.original_cuda_env = None
        self.model_arch_table = model_arch_table
        self.source_table = source_table
        self.validation_table = validation_table
        self.models_root = models_root
        self.logs_root = logs_root
        if CUDA_VISIBLE_DEVICES_KEY in os.environ:
            self.original_cuda_env = os.environ[CUDA_VISIBLE_DEVICES_KEY]

        self.mst_catlog = {}
        self.init_catlogs()
        self.init_sess()

    def init_sess(self):
        self.manager = Manager()
        self.model_keys = []
        self.model_info_ordered = defaultdict(list)
        self.return_dict_model = self.manager.dict()
        self.return_dict_grand = {}
        self.model_dirs = {}
        self.model_configs = {}

    def init_epoch(self):
        self.return_dict_job = self.manager.dict()
        self.procs = {}
        self.model_dist_pairs = [(i, j) for i in self.model_keys
                                 for j in self.dist_keys]
        random.shuffle(self.model_dist_pairs)
        self.model_states = {i: False for i in self.model_keys}
        self.dist_states = {i: False for i in self.dist_keys}
        self.model_on_dist = {dist_key: -1 for dist_key in self.dist_keys}

        for job_key in self.model_dist_pairs:
            self.return_dict_job[job_key] = {
                'status': None
            }

    def run(self):
        self.load_msts()
        if self.logs_root:
            model_info_filepath = os.path.join(
                self.logs_root, "models_info.pkl")
            jobs_info_filepath = os.path.join(self.logs_root, "jobs_info.pkl")
        for epoch in range(1, self.epochs + 1):
            self.init_epoch()
            logs("EPOCH:{}".format(epoch))
            self.train_one_epoch(epoch)
            self.return_dict_grand[epoch] = dict(self.return_dict_job)
            if self.logs_root:
                with open(model_info_filepath, "wb") as f:
                    dill.dump(self.model_info_ordered, f)
                with open(jobs_info_filepath, "wb") as f:
                    dill.dump(self.return_dict_grand, f)

    def init_catlogs(self):

        if self.use_gpus:
            self.accessible_gpus_for_seg = self.db.get_accessible_gpus_for_seg(
                self.schema_madlib, self.segments_per_host, MODULE_NAME)
        else:
            self.accessible_gpus_for_seg = self.db.get_seg_number() * [0]
        # Compute total images on each segment
        self.dist_key_mapping, self.images_per_seg_train = \
            self.db.get_image_count_per_seg_for_minibatched_data_from_db(
                self.source_table)
        self.dist_keys = self.dist_key_mapping
        if self.validation_table:
            self.seg_ids_val, self.images_per_seg_valid = \
                self.db.get_image_count_per_seg_for_minibatched_data_from_db(
                    self.validation_table)

    def fetch_model_from_params(self, compile_params, fit_params, model_id):
        model_arch, model_weights = self.db.get_model_arch_weights(
            self.model_arch_table, model_id)

        serialized_weights = get_initial_weights(None, model_arch,
                                                 model_weights, False,
                                                 self.use_gpus,
                                                 self.accessible_gpus_for_seg)
        return model_arch, serialized_weights

    def mst_to_model(self, mst):
        model_key = mst_2_str(mst)
        logs("Loading models, {}".format(model_key))
        model = create_model_from_mst(mst)
        compile_params, fit_params = params_fac(mst)
        model_arch = model.to_json()
        serialized_weights = get_initial_weights(None, model_arch, None, None,
                                                 None, None)
        return model_key, \
            model_arch, compile_params, fit_params, serialized_weights

    def load_msts(self):
        for mst in self.msts:
            model_key, \
                model_arch, \
                compile_params, \
                fit_params, \
                serialized_weights = self.mst_to_model(mst)
            self.return_dict_model[model_key] = serialized_weights
            serialized_weights_dir = os.path.join(self.models_root, model_key)
            self.model_dirs[model_key] = serialized_weights_dir
            with open(serialized_weights_dir, 'wb') as f:
                f.write(serialized_weights)
            self.model_keys.append(model_key)
            self.model_configs[
                model_key] = model_arch, compile_params, fit_params

    def train_on_worker(self,
                        dist_key,
                        model_arch,
                        compile_params,
                        fit_params,
                        serialized_weights_dir,
                        epoch,
                        job_key=None,
                        model_key=None,
                        return_dict_job=None,
                        return_dict_model=None):
        failed = False
        try:
            begin = time.time()
            timestamp_begin = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")
            ctq = ConcurrentTargetedQueryService(
                self.db_creds, self.schema_madlib, dist_key,
                self.dist_key_mapping, self.seg_ids_val, model_arch,
                compile_params, fit_params, serialized_weights_dir,
                self.accessible_gpus_for_seg, self.images_per_seg_train,
                self.images_per_seg_valid, self.source_table,
                self.validation_table, self.use_gpus)
            ctq.train_prepare()
            ctq.valid_prepare()
            init_end = time.time()
            serialized_weights = ctq.train()
            with open(serialized_weights_dir, 'wb') as f:
                f.write(serialized_weights)
            loss_train, metric_train = ctq.valid(on_train=True)
            train_end = time.time()
            loss_valid, metric_valid = ctq.valid(on_train=False)
            valid_end = time.time()
            
        except Exception:
            failed = True
            traceback.print_exc()
            raise Exception("FATAL ERROR")
        if return_dict_job is not None and return_dict_model is not None:
            if job_key in return_dict_job and return_dict_job[job_key][
                    'status'] is not None:
                logs("Status: {}".format(return_dict_job[job_key]['status']))
                raise Exception("Job key already processed!")
            status = "SUCCESS" if not failed else "FAILED"
            if failed:
                return_dict_job[job_key]['status'] = status
                return_dict_model[model_key] = None
            else:
                return_dict_model[model_key] = serialized_weights
                final_end = time.time()
                timestamp_end = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
                return_dict_job[job_key] = {
                    'status': status,
                    'epoch': epoch,
                    'dist_key': dist_key,
                    'model_key': model_key,
                    'loss_train': loss_train,
                    'metric_train': metric_train,
                    'loss_valid': loss_valid,
                    'metric_valid': metric_valid,
                    'start_time': timestamp_begin,
                    'end_time': timestamp_end,
                    'init_time': init_end - begin,
                    'train_time': train_end - init_end,
                    'valid_time': valid_end - train_end,
                    'exit_time': final_end - valid_end
                }
        return status, serialized_weights, \
            loss_train, metric_train, loss_valid, metric_valid

    def _get_runnable_model(self, target_dist_key, model_dist_pairs,
                            model_states):
        for model_key, dist_key in model_dist_pairs:
            # worker matches and model idle|
            if dist_key == target_dist_key and not model_states[model_key]:
                return model_key
        return -1

    def assign_one_model_to_dist(self, model_key, dist_key, epoch):
        job_key = (model_key, dist_key)
        serialized_weights_dir = self.model_dirs[model_key]
        model_arch, compile_params, fit_params = self.model_configs[model_key]
        proc = Process(target=self.train_on_worker,
                       args=[
                           dist_key, model_arch, compile_params, fit_params,
                           serialized_weights_dir, epoch, job_key, model_key,
                           self.return_dict_job, self.return_dict_model
                       ])
        self.procs[job_key] = proc
        proc.start()
        self.model_states[model_key] = True
        self.dist_states[dist_key] = True
        self.model_on_dist[dist_key] = model_key
        # proc.join()

    def peek_job(self, model_key, dist_key):
        job_key = (model_key, dist_key)
        proc = self.procs[job_key]
        job_dict = self.return_dict_job[job_key]
        status = job_dict['status']
        if status == 'SUCCESS' and not proc.is_alive():
            # sub-epoch completed
            self.model_dist_pairs.remove(job_key)
            self.model_states[model_key] = False
            self.dist_states[dist_key] = False
            self.model_on_dist[dist_key] = -1
            self.model_info_ordered[model_key].append(job_dict)
            proc.terminate()
            logs("JOBS DONE: {}".format(job_key))
            logs("LEFT JOBS: {}".format(len(self.model_dist_pairs)))
        elif status == 'FAILED':
            raise Exception("Fatal error!")

    def train_one_epoch(self, epoch):
        while len(self.model_dist_pairs) > 0:
            for dist_key in self.dist_keys:
                if not self.dist_states[dist_key]:
                    #                     Assign to idle worker
                    model_key = self._get_runnable_model(
                        dist_key, self.model_dist_pairs, self.model_states)
                    if model_key != -1:
                        job_key = (model_key, dist_key)
                        logs("JOBS ALLOCATING: {}".format(job_key))
                        self.assign_one_model_to_dist(
                            model_key, dist_key, epoch)
                        logs("JOBS ALLOCATED: {}".format(job_key))
                else:
                    #             Peek completed
                    model_key = self.model_on_dist[dist_key]
                    if model_key != -1:
                        self.peek_job(model_key, dist_key)


if __name__ == '__main__':
    args, msts = main_prepare()
    if args.run:
        logs("START RUNNING CTQ")
        set_seed(SEED)
        print(args.train_name, args.valid_name)
        ctq_client = ConcurrentTargetedQueryClient(
            db_creds=cats,
            schema_madlib='madlib',
            msts=msts,
            source_table=args.train_name,
            validation_table=args.valid_name,
            model_arch_table='model_arch_library',
            use_gpus=not args.criteo,
            epochs=args.num_epochs,
            logs_root=args.logs_root,
            models_root=args.models_root
        )
        ctq_client.run()
        summary = get_summary(ctq_client.model_info_ordered)
        print(summary)
        logs("END RUNNING")



