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



def convert_to_float(lists):
  return [float(el) if not isinstance(el,list) else convert_to_float(el) for el in lists]

def parser(line):
    parsed = line.strip('\n').split('|')
    record_id = parsed[0]
    feature = literal_eval(parsed[1].replace('{', '[').replace('}', ']'))
    # feature = convert_to_float(feature)
    feature = np.asarray(feature, dtype=np.float32)
    label = int(parsed[2])
    return record_id, feature, label

def main(file_list, output_list, outformat='h5'):
    logs("Start")
    for purpose in ['valid', 'train']:
        logs("Starting {}".format(purpose))
        filename = file_list[purpose]
        ofilename = output_list[purpose]
        logs("Start loading into memory {}".format(purpose))
        with open(filename) as f_in:
            lines = f_in.readlines()
        logs("End loading into memory {}".format(purpose))
        pool = Pool(processes=36)
        features_lst = []
        labels_lst = []
        logs("Start parsing {}".format(purpose))
        for res in tqdm.tqdm(pool.imap_unordered(parser, lines), total=len(lines)):
            _, feature, label = res
            features_lst.append(feature) 
            labels_lst.append(label) 
        pool.close()
        pool.join()
        logs("End parsing {}".format(purpose))
        logs("Start typecasting {}".format(purpose))
        np_arr_features = np.asarray(features_lst)
        np_arr_labels = np.asarray(labels_lst)
        logs("End typecasting {}".format(purpose))
        if outformat == 'h5':
            logs("Start writing {}".format(purpose))
            hf = h5py.File(ofilename, 'w')
            hf.create_dataset('features', data=np_arr_features)
            hf.create_dataset('labels', data=np_arr_labels)
            hf.close()
            logs("End writing {}".format(purpose))
            logs("Ending {}".format(purpose))
        else:
            np_arr = np.hstack([np_arr_features, np_arr_labels[:, None]])
            np.save(ofilename, np_arr)
    logs("Ending")

if __name__ == '__main__':
    file_list = {'train':'/mnt/nfs/hdd/criteo/imagenet_train_0.out', 
                'valid':'/mnt/nfs/hdd/criteo/imagenet_valid_0.out'}

    output_list = {'train':'/mnt/nfs/imagenet_train_0.h5', 
                'valid':'/mnt/nfs/imagenet_valid_0.h5'}
    main(file_list, output_list)