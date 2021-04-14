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
import os
import numpy as np
import dill
import torch
from cerebro_gpdb.utils import cats
from cerebro_gpdb.utils import logs
from cerebro_gpdb.utils import logsc
from cerebro_gpdb.da import DirectAccessClient
from cerebro_gpdb.da import input_fn
from cerebro_gpdb.run_pytorchddp import TRAIN
from cerebro_gpdb.run_pytorchddp import VALID
from cerebro_gpdb.run_pytorchddp import TorchTrainer
from cerebro_gpdb.run_pytorchddp import init_ddp
from cerebro_gpdb.in_rdbms_helper import main_prepare

IMAGENET_COUNT_PER_PAR = 160160
CRITEO_COUNT_PER_PAR = 1624157
DEBUG = False
if DEBUG:
    IMAGENET_COUNT_PER_PAR /= 100
RANK_FILE_PATH = '/mnt/nfs/rank_file.dill'


class DADataset(torch.utils.data.Dataset):
    def __init__(self, file_path, dataset_name='imagenet', mode=TRAIN, **kwargs):
        self.df_actual_data = input_fn(file_path, **kwargs)
        self.mode = mode
        self.dataset_name = dataset_name
        if dataset_name == 'imagenet':
            self.length = IMAGENET_COUNT_PER_PAR
        elif dataset_name == 'criteo':
            self.length = CRITEO_COUNT_PER_PAR
        self.build_index()

    def __getitem__(self, i):  # to enable indexing
        if self.mode == TRAIN and i not in self.index:
            i = i % self.actual_length
        buffer_id, idx = self.index[i]
        label = np.argmax(self.df_actual_data[buffer_id]['dependent_var'][idx])
        image = self.df_actual_data[buffer_id]['independent_var'][idx]
        if self.dataset_name == 'imagenet':
            image = np.rollaxis(image, 2, 0)
        return (
            image,
            label,
        )

    def __len__(self):
        return self.length

    def build_index(self):
        self.index = {}
        self.actual_length = 0
        for buffer_id, buffer in self.df_actual_data.items():
            dependent_var = buffer['dependent_var']
            independent_var = buffer['independent_var']
            assert dependent_var.shape[0] == independent_var.shape[0]
            len_buffer = dependent_var.shape[0]
            for i in range(len_buffer):
                self.index[self.actual_length] = (buffer_id, i)
                self.actual_length += 1
        if self.mode != TRAIN:
            self.length = self.actual_length


class TorchTrainerDA(TorchTrainer):
    def preload_data_helper(self, partition, phases):
        with logsc(
            'loading data, partition-{}, loading:{}, dirs:{}'.format(
                partition, phases, self.file_paths)
        ):
            datasets = {
                phase: DADataset(self.file_paths[phase][partition],
                                 dataset_name=self.dataset_name, mode=phase)
                for phase in phases
            }
            dataloaders = self.prepare_dataloaders(datasets, phases)
        return datasets, dataloaders


def client_main():
    da = DirectAccessClient(
        cats, args.db_name, args.train_name, args.valid_name, args.size)
    data_cat, sys_cats = da.generate_cats()
    rank_filepath = {}
    data_root = data_cat['data_root']
    for rank in range(len(da.workers)):
        train_path = os.path.join(data_root, data_cat['train'][rank])
        valid_path = os.path.join(data_root, data_cat['valid'][rank])
        rank_filepath[rank] = [train_path, valid_path]

    with open(RANK_FILE_PATH, "wb") as f:
        dill.dump(rank_filepath, f)


def get_file_paths_da(rank):
    with open(RANK_FILE_PATH, "rb") as f:
        rank_filepath = dill.load(f)
    train_path, valid_path = rank_filepath[rank]
    if DEBUG:
        file_paths = {
            TRAIN: [valid_path],
            VALID: [valid_path]
        }
    else:
        file_paths = {
            TRAIN: [train_path],
            VALID: [valid_path]
        }
    return file_paths


def main(args, msts, get_file_path_fn, Trainer, rank):
    gpu, dataset_name, _ = init_ddp(args, rank)

    # world_size = args.size
    # torch.distributed.init_process_group(
    #     backend='nccl',
    #     init_method='tcp://worker0:23456',
    #     rank=rank, world_size=world_size)
    file_paths = get_file_path_fn(rank)
    print(torch.distributed.is_initialized())
    # dataset_name = 'criteo' if args.criteo else 'imagenet'
    trainer = Trainer(
        msts[0],
        file_paths,
        gpu=gpu,
        persist=True, dist=True, dataset_name=dataset_name)
    for i, mst in enumerate(msts):
        with logsc('MST #{}: {}'.format(i, mst)):
            trainer.update_model(mst)
            if DEBUG:
                args.num_epochs = 2
            all_logs = trainer.train(0, args.num_epochs)
            logs(all_logs)
            with open(os.path.join(
                args.logs_root, 'all_logs_rank{}_index{}.pkl'.format(
                    rank, i)
            ), "wb") as f:
                dill.dump(all_logs, f)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args, msts = main_prepare(shuffle=False, backend='pytorch', verbose=True)
    print(args.train_name, args.valid_name)
    if args.client:
        client_main()
    elif args.worker:
        RANK = int(os.getenv('WORKER_NUMBER'))
        main(args, msts, get_file_paths_da, TorchTrainerDA, RANK)
