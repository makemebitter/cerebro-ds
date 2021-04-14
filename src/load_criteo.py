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
from load_imagenet import Loader
from load_imagenet import loader_args
from load_imagenet import get_all
import numpy as np
import glob
import os
from utils import cats_criteo as cats


class CriteoLoader(Loader):
    def __init__(
        self,
        db_creds,
        num_workers=1,
        size_scalability=None,
        train_buffer_size='NULL'
    ):
        super(CriteoLoader, self).__init__(
            db_creds, num_workers, size_scalability, train_buffer_size)
        self.size_scalability = size_scalability
        self.segments_to_use = \
            'segments_to_use_{}'.format(size_scalability) if \
            self.size_scalability is not None else 'all_segments'

    def get_data_label(self, file_path):
        np_arr = np.load(file_path)
        np_images = np_arr[:, :-1].astype(np.float32)
        np_labels = np_arr[:, -1].astype(int)
        return np_images, np_labels


if __name__ == "__main__":
    args, db_creds = loader_args()
    criteo_loader = CriteoLoader(
        db_creds, 16, size_scalability=args.size_scalability)

    name_list = ['criteo_train_data', 'criteo_valid_data']
    purpose_list = ['train', 'valid']
    if args.load:
        print("START LOADING")
        file_list_list = [get_all(
            cats.train_root, 'npy'), get_all(cats.valid_root, 'npy')]
        for name, file_list in zip(name_list, file_list_list):
            criteo_loader.drop_table(name)
            criteo_loader.load_many(file_list, name)
        print("END LOADING")
    if args.pack:
        print("START PACKING")
        criteo_loader.create_binding()
        train_name, valid_name = name_list
        train_packed_name = criteo_loader.get_pack_name(train_name)[0]
        for purpose in purpose_list:
            print("START PACKING: {}".format(purpose))
            criteo_loader.pack(train_name, purpose,
                               valid_name, train_packed_name)
            print("END PACKING: {}".format(purpose))
        print("END PACKING")
