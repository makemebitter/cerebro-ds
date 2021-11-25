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
from cerebro_gpdb.in_rdbms_helper import main_prepare
from cerebro_gpdb.pathmagic import *
import sys
import os
import gc
import torch
import torchvision
from torchvision import models
from torch.nn.parallel import DistributedDataParallel as DDP
from cerebro_gpdb.run_cerebro_standalone_helper import input_fn
from cerebro_gpdb.run_cerebro_standalone_helper import input_fn_criteo
from cerebro_gpdb.run_cerebro_standalone_helper import DATA_CATALOG
from cerebro_gpdb.criteocat import INPUT_SHAPE as INPUT_SHAPE_CRITEO
from cerebro_gpdb.criteocat import NUM_CLASSES as NUM_CLASSES_CRITEO
from collections import OrderedDict
from cerebro_gpdb.utils import logs
from cerebro_gpdb.utils import logsc
import numpy as np
import glob
import dill
IMAGENET = 'imagenet'
CRITEO = 'criteo'
TRAIN = 'train'
VALID = 'valid'
PHASES = [TRAIN, VALID]
# MASTER = 'M'
# WORKER = 'W'
# ROLE = MASTER if 'master' == os.getenv('WORKER_NAME') else WORKER
# if ROLE == MASTER:
#     RANK = 0
# else:


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, batch_size, shuffle, *arrays):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        tensors = [torch.from_numpy(arr) for arr in arrays]
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def next(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        data = input_fn(file_path, one_hot=False)
        self.images = np.concatenate([x['images'] for x in data], axis=0)
        #         channel last to channel first
        self.images = np.rollaxis(self.images, 3, 1)
        self.labels = np.concatenate([x['labels'] for x in data], axis=0)
        assert self.images.shape[0] == self.labels.shape[0]
        self.length = self.images.shape[0]
        del data

    def __getitem__(self, index):  # to enable indexing
        image = self.images[index]
        label = self.labels[index]
        return (
            image,
            label,
        )

    def __len__(self):
        return self.length


class NPYDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.data = input_fn_criteo(file_path, one_hot=False)
        assert self.data['labels'].shape[0] == self.data['images'].shape[0]
        self.length = self.data['labels'].shape[0]

    def __getitem__(self, index):  # to enable indexing
        image = self.data['images'][index]
        label = self.data['labels'][index]
        return (
            image,
            label,
        )

    def __len__(self):
        return self.length


def add_metrics_to_log(log, metrics, y_true, y_pred, prefix=''):
    for metric in metrics:
        q = metric(y_true, y_pred)
        log[prefix + metric.__name__] = q
    return log


def log_to_message(log, precision=4):
    fmt = "{0}: {1:." + str(precision) + "f}"
    return "    ".join(fmt.format(k, v) for k, v in log.items())


class ProgressBar(object):
    """Cheers @ajratner"""

    def __init__(self, n, length=40):
        # Protect against division by zero
        self.n = max(1, n)
        self.nf = float(n)
        self.length = length
        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i / 100.0 * n) for i in range(101)])
        self.ticks.add(n - 1)
        self.bar(0)

    def bar(self, i, message=""):
        """Assumes i ranges through [0, n-1]"""
#         if i in self.ticks:
        b = int(np.ceil(((i + 1) / self.nf) * self.length))
        sys.stdout.write("\r[{0}{1}] {2}% {3}/{4} \t{5}".format(
            "=" * b, " " * (self.length - b),
            int(100 * ((i + 1) / self.nf)), i + 1, int(self.nf), message))
        sys.stdout.flush()

    def close(self, message=""):
        # Move the bar to 100% before closing
        self.bar(self.n - 1)
        sys.stdout.write("\n{0}\n\n".format(message))
        sys.stdout.flush()


def accuracy(output, target, topk=(1, ), binary=False):
    """Computes the precision@k for the specified values of k"""
    if binary:
        batch_size = target.size(0)
        _, pred = torch.max(output.data, 1)
        correct = (pred == target).sum().item()
        res = [torch.tensor(correct / batch_size)]
    else:
        maxk = max(topk)
        maxk = min(maxk, output.shape[1])
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(1.0 / batch_size))
    return res


class TorchTrainer(object):
    def __init__(
            self,
            mst,
            file_paths,
            gpu=True,
            persist=False,
            dist=False,
            dataset_name=IMAGENET):
        self.dataset_name = dataset_name
        self.gpu = gpu
        if self.gpu:
            self.num_data_workers = 32
        else:
            self.num_data_workers = 16
        self.persist = persist
        self.file_paths = file_paths
        assert len(file_paths[TRAIN]) == len(file_paths[VALID])
        self.total_partitions = len(file_paths[TRAIN])
        self.device = self.get_device()
        self.dist = dist
        self.update_model(mst)
#         self.datasets, self.dataloaders = self.preload_data()
        self.persisted_data = {}

    def clear_model(self):
        self.model = None
        gc.collect()
        if self.gpu:
            torch.cuda.empty_cache()

    def update_model(self, mst):
        self.clear_model()
        self.mst = mst
        self.model, self.criterion, self.log_softmax, self.optimizer = self.init_model()
        if self.dist:
            self.init_dist()

    def preload_data_helper(self, partition, phases):
        raise NotImplementedError

    def preload_data(self, partition=0, phase='train', both=False):
        phases = []
        if phase == 'train':
            phases.append(TRAIN)
        else:
            phases.append(VALID)
        if both:
            phases = PHASES

        datasets = {}
        dataloaders = {}
        if self.persist:
            for phase in phases:
                if (partition, phase) in self.persisted_data:
                    datasets[phase], _ = self.persisted_data[(
                        partition, phase)]
                    # force to rebuild dataloaders
                    dataloaders[phase] = self.prepare_dataloaders(
                        datasets, [phase])[phase]
                    self.persisted_data[(
                        partition, phase)] = datasets[phase], dataloaders[phase]
                else:
                    #                     load and cache
                    datasets_loaded, dataloader_loaded = \
                        self.preload_data_helper(
                            partition, [phase])
                    self.persisted_data[(
                        partition, phase)] = \
                        datasets_loaded[phase], dataloader_loaded[phase]
                    datasets[phase], dataloaders[phase] = self.persisted_data[(
                        partition, phase)]
        else:
            datasets, dataloaders = self.preload_data_helper(partition, [
                                                             phase])

        return datasets, dataloaders

    def init_dist(self):
        self.model = DDP(self.model)

    def init_model(self):
        with logsc('initilizing model'):
            model = self.model_fac()
            criterion = torch.nn.CrossEntropyLoss()
            log_softmax = torch.nn.LogSoftmax()
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=self.mst['learning_rate'],
                                         weight_decay=self.mst['lambda_value'])
        return model, criterion, log_softmax, optimizer

    def model_fac(self):
        if self.mst['model'] == "resnet50":
            model = models.resnet50(pretrained=False, progress=True)
        elif self.mst['model'] == "vgg16":
            model = models.vgg16(pretrained=False, progress=True)
        elif self.mst['model'] == 'confA':
            model = torch.nn.Sequential(
                torch.nn.Linear(INPUT_SHAPE_CRITEO[0], 1000),
                torch.nn.ReLU(),
                torch.nn.Linear(1000, 500),
                torch.nn.ReLU(),
                torch.nn.Linear(500, NUM_CLASSES_CRITEO)
            )
        model = model.to(self.device)
        return model

    def minibatch_train(self, i, data, sublog, phase):
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        # curr_batch_size = labels.shape[0]
        outputs, loss = self.minibatch_train_helper(phase, inputs, labels)
        outputs_softmax = self.log_softmax(outputs)
        if self.dataset_name == IMAGENET:
            top_1_acc, top_5_acc = accuracy(outputs_softmax, labels, (1, 5))
        elif self.dataset_name == CRITEO:
            top_1_acc = accuracy(outputs_softmax, labels, None, binary=True)[0]
            top_5_acc = torch.tensor(1)
        self.epoch_loss += loss.item()
        self.epoch_top_1_acc += top_1_acc.item()
        self.epoch_top_5_acc += top_5_acc.item()
        sublog['{}_loss'.format(phase)] = self.epoch_loss / (i + 1)
        sublog['{}_acc_top1'.format(
            phase)] = self.epoch_top_1_acc / (i + 1)
        sublog['{}_acc_top5'.format(
            phase)] = self.epoch_top_5_acc / (i + 1)
        return sublog

    def minibatch_train_helper(self, phase, inputs, labels):
        if phase == TRAIN:
            self.model.train()
            # zero the parameter gradients
            # with logsc('zero_grad'):
            self.optimizer.zero_grad()
            # forward + backward + optimize
            # with logsc('forward'):
            outputs = self.model(inputs)
            # with logsc('cal_loss'):
            loss = self.criterion(outputs, labels)
            # with logsc('backward'):
            loss.backward()
            # with logsc('optimize'):
            self.optimizer.step()
        elif phase == VALID:
            self.model.eval()
            with torch.no_grad():
                # with logsc('forward'):
                outputs = self.model(inputs)
                # with logsc('cal_loss'):
                loss = self.criterion(outputs, labels)
        return outputs, loss

    def get_device(self):
        if torch.cuda.is_available() and self.gpu:
            device = torch.device("cuda:0")
            device_ids = list(range(torch.cuda.device_count()))
            gpus = len(device_ids)
            print('GPU detected')
        else:
            device = torch.device("cpu")
            print('No GPU or GPU disabled. switching to CPU')
        return device

    def train(self, initial_epoch, epochs):
        all_logs = OrderedDict()
        for epoch in range(initial_epoch,
                           epochs):  # loop over the dataset multiple times
            logs("Epoch {0} / {1}".format(epoch + 1, epochs))
            log = OrderedDict()

            for phase in PHASES:
                with logsc('training-{}'.format(phase)):
                    sublog = OrderedDict()
                    print("phase: {}".format(phase))
                    for partition in range(self.total_partitions):
                        both = self.persist
                        _, dataloaders = self.preload_data(
                            partition, phase, both)
                        dataloader = dataloaders[phase]
                        pb = ProgressBar(len(dataloader))
                        self.epoch_loss = 0
                        self.epoch_top_1_acc = 0
                        self.epoch_top_5_acc = 0
                        for i, data in enumerate(dataloader, 0):
                            sublog = self.minibatch_train(
                                i, data, sublog, phase)
                            pb.bar(i, log_to_message(sublog))
                    pb.close(log_to_message(sublog))
                    log[phase] = sublog
            all_logs[epoch] = log
        return all_logs

    def prepare_dataloaders(self, datasets, phases):
        if self.dataset_name == CRITEO:
            # dataloaders = {
            #     phase: FastTensorDataLoader(
            #         self.mst['batch_size'],
            #         False,
            #         datasets[phase].data['images'],
            #         datasets[phase].data['labels']
            #     )
            #     for phase in phases
            # }
            dataloaders = {
                phase: torch.utils.data.DataLoader(
                    datasets[phase],
                    batch_size=self.mst['batch_size'],
                    shuffle=False,
                    num_workers=self.num_data_workers)
                for phase in phases
            }
        elif self.dataset_name == IMAGENET:
            dataloaders = {
                phase: torch.utils.data.DataLoader(
                    datasets[phase],
                    batch_size=self.mst['batch_size'],
                    shuffle=False,
                    num_workers=self.num_data_workers)
                for phase in phases
            }
        return dataloaders


class TorchTrainerFilesystem(TorchTrainer):
    def preload_data_helper(self, partition, phases):
        if self.dataset_name == IMAGENET:
            Dataset = HDF5Dataset
        elif self.dataset_name == CRITEO:
            Dataset = NPYDataset
        with logsc(
            'loading data, partition-{}, loading:{}, dirs:{}'.format(
                partition, phases, self.file_paths)
        ):
            datasets = {
                phase: Dataset(self.file_paths[phase][partition])
                for phase in phases
            }
            dataloaders = self.prepare_dataloaders(datasets, phases)
        return datasets, dataloaders


def get_file_paths_filesystem(rank=None):
    datacat = DATA_CATALOG['imagenet_cerebro_standalone']
    data_root = datacat['data_root']
    train_files = os.path.join(data_root, 'train', '*.h5')
    valid_files = os.path.join(data_root, 'valid', '*.h5')
    file_paths = {
        TRAIN: glob.glob(train_files),
        VALID: glob.glob(valid_files)
    }
    return file_paths


def get_file_paths_filesystem_criteo(rank=None):
    datacat = DATA_CATALOG['criteo_cerebro_standalone']
    data_root = datacat['data_root']
    train_files = os.path.join(data_root, 'train', '*.npy')
    valid_files = os.path.join(data_root, 'valid', '*.npy')
    file_paths = {
        TRAIN: glob.glob(train_files),
        VALID: glob.glob(valid_files)
    }
    return file_paths


def train_one_mst(mst, get_file_path_fn, Trainer, rank, **kwargs):
    with logsc('MST #{}: {}'.format(args.single_mst_index, mst)):
        file_paths = get_file_path_fn(rank)
        print(torch.distributed.is_initialized())
        trainer = Trainer(
            mst,
            file_paths,
            persist=True, dist=True, **kwargs)
        all_logs = trainer.train(0, args.num_epochs)
        logs(all_logs)
        with open(os.path.join(
            args.logs_root, 'all_logs_rank{}_index{}.pkl'.format(
                rank, args.single_mst_index)
        ), "wb") as f:
            dill.dump(all_logs, f)


def init_ddp(args, rank):
    world_size = args.size
    if args.criteo:
        gpu = False
        dataset_name = CRITEO
        get_file_path_fn = get_file_paths_filesystem_criteo
        backend = 'gloo'
    else:
        gpu = True
        dataset_name = IMAGENET
        get_file_path_fn = get_file_paths_filesystem
        backend = 'nccl'
    torch.distributed.init_process_group(
        backend=backend,
        init_method='tcp://worker0:23456',
        rank=rank, world_size=world_size)

    return gpu, dataset_name, get_file_path_fn


def main(args, msts, Trainer, rank):
    assert len(msts) == 1
    mst = msts[0]
    gpu, dataset_name, get_file_path_fn = init_ddp(args, rank)
    train_one_mst(
        mst, get_file_path_fn, Trainer, rank, gpu=gpu, dataset_name=dataset_name)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    RANK = int(os.getenv('WORKER_NUMBER'))
    args, msts = main_prepare(shuffle=False, backend='pytorch', verbose=True)
    main(args, msts, TorchTrainerFilesystem, RANK)
