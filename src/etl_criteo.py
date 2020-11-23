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

import numpy as np
import pandas as pd
from ast import literal_eval
import h5py
from multiprocessing import Pool
import tqdm
from utils import logs
from etl_imagenet import main


def convert_to_float(lists):
  return [float(el) if not isinstance(el,list) else convert_to_float(el) for el in lists]

def parser(line):
    line = line.strip('\n')
    parsed = line.split('|')
    parsed = line.split('|')
    record_id = parsed[0]
    feature = literal_eval(parsed[1].replace('{', '[').replace('}', ']'))
    feature = convert_to_float(feature)
    label = parsed[2]
    label = [int(x) for x in label]
    return record_id, feature, label

if __name__ == '__main__':
    file_list = {'train':'/mnt/nfs/hdd/criteo/unload/train_0.out', 
            'valid':'/mnt/nfs/hdd/criteo/unload/valid_0.out'}

    output_list = {'train':'/mnt/nfs/hdd/criteo/unload/train_0.npy', 
            'valid':'/mnt/nfs/hdd/criteo/unload/valid_0.npy'}
    main(file_list, output_list)