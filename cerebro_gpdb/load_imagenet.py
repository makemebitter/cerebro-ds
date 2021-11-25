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
import sys
from math import ceil
import argparse
import numpy as np
import h5py
from madlib_image_loader import ImageLoader, DbCredentials
import os
import glob
from utils import DBBase
from utils import cats_imagenet as cats


TOAST_LIMIT_BYTES = 1073741824
ONE_IMAGE_BYTES = 112 * 112 * 3 * 4

VALID_BUFFER_SIZE = int(ceil(50000 / 16))
TRAIN_BUFFER_SIZE = 3210
SEGMENTS_TO_USE = 'segments_to_use'

# sys.path.insert(0, '/home/gpadmin/.local/lib/python3.5/site-packages/')
print(sys.path)

try:
    sys.path.remove('/usr/local/gpdb/lib/python')
except Exception:
    pass


def get_all(fdir, ext='h5'):
    return sorted(glob.glob(os.path.join(fdir, '*.{}'.format(ext))))


class Loader(DBBase):
    def __init__(
        self,
        db_creds,
        num_workers=1,
        size_scalability=None,
        train_buffer_size=None
    ):
        super(Loader, self).__init__(db_creds)
        self.iloader = ImageLoader(num_workers=num_workers, db_creds=db_creds)
        self.size_scalability = size_scalability
        self.train_buffer_size = train_buffer_size
        self.segment_list_limit_offset_map = {
            1: (1, 0),
            2: (2, 1),
            4: (4, 3),
            6: (6, 2)
        }

    def get_data_label(self, file_path):
        raise NotImplementedError()

    def load_one(self, file_path, name, force=False):
        print("Loading {}".format(file_path))
        exists = self.if_exists_table(name)
        if exists and not force:
            raise Exception("Table {} already exists!".format(name))
        np_images, np_labels = self.get_data_label(file_path)
        self.iloader.load_dataset_from_np(
            np_images, np_labels, name, append=exists)

    def load_many(self, file_list, name, force=False):
        exists = self.if_exists_table(name)
        if exists and not force:
            raise Exception("Table {} already exists!".format(name))
        for file_path in file_list:
            self.load_one(file_path, name, True)

    def create_binding(self):
        segments_to_use = SEGMENTS_TO_USE
        self.cursor.execute("""
            DROP TABLE IF EXISTS host_gpu_mapping_tf;
            SELECT * FROM madlib.gpu_configuration('host_gpu_mapping_tf');
            SELECT * FROM host_gpu_mapping_tf ORDER BY hostname, gpu_descr;
            DROP TABLE IF EXISTS {segments_to_use};
            CREATE TABLE {segments_to_use} AS
                SELECT DISTINCT dbid, hostname 
                FROM gp_segment_configuration JOIN host_gpu_mapping_tf USING (hostname)
                WHERE role='p' AND content>=0;
            """.format(**locals())
        )
        if self.size_scalability is not None:
            limit, offset = self.segment_list_limit_offset_map[
                self.size_scalability]
            self.cursor.execute("""
            DROP TABLE IF EXISTS {self.segments_to_use};
            CREATE TABLE {self.segments_to_use} AS
                select * from {segments_to_use} order by (dbid) limit {limit} offset {offset};
            """.format(**locals())
            )

    def get_pack_name(self, name):
        if self.size_scalability is not None:
            name_packed = name + '_packed_{}'.format(self.size_scalability)
            name_packed_summary = name_packed + \
                '_summary'
        else:
            name_packed = name + '_packed'
            name_packed_summary = name_packed + '_summary'
        return name_packed, name_packed_summary

    def pack_train(self, name):
        name_packed, name_packed_summary = self.get_pack_name(name)
        function = 'training_preprocessor_dl'

        sql_query = """
            DROP TABLE IF EXISTS {name_packed}, {name_packed_summary};
            SELECT madlib.{function}(
                '{name}',        -- Source table
                '{name_packed}', -- Output table
                'y',                    -- Dependent variable
                'x',                    -- Independent variable
                {self.train_buffer_size},                  -- Buffer size
                1.0,                 -- Normalizing constant
                NULL,                  -- Number of classes
                '{self.segments_to_use}'  -- Distribution rules
            ); """.format(**locals())
        print(sql_query)
        self.cursor.execute(sql_query)

    def pack_valid(self, name, train_packed_name):
        name_packed, name_packed_summary = self.get_pack_name(name)
        function = 'validation_preprocessor_dl'
        valid_buffer_size = self.train_buffer_size
        sql_query = """
            DROP TABLE IF EXISTS {name_packed}, {name_packed_summary};
            SELECT madlib.{function}(
                '{name}',        -- Source table
                '{name_packed}', -- Output table
                'y',                    -- Dependent variable
                'x',                    -- Independent variable
                '{train_packed_name}',
                {valid_buffer_size},                  -- Buffer size
                '{self.segments_to_use}'  -- Distribution rules
            ); """.format(**locals())
        print(sql_query)
        self.cursor.execute(sql_query)

    def pack(self, train_name, purpose, valid_name=None,
             train_packed_name=None):
        if purpose == 'train':
            self.pack_train(train_name)
        elif purpose == 'valid':
            self.pack_valid(valid_name, train_packed_name)


class ImageNetLoader(Loader):
    def __init__(
        self,
        db_creds,
        num_workers=1,
        size_scalability=None,
        train_buffer_size=TRAIN_BUFFER_SIZE,
        no_gpu=False
    ):
        super(ImageNetLoader, self).__init__(
            db_creds,
            num_workers=num_workers,
            size_scalability=size_scalability,
            train_buffer_size=train_buffer_size)
        if no_gpu:
            self.segments_to_use = 'all_segments'
        else:
            if size_scalability is not None:
                self.segments_to_use = '{}_{}'.format(
                    SEGMENTS_TO_USE, size_scalability)
            else:
                self.segments_to_use = SEGMENTS_TO_USE

    def get_data_label(self, file_path):
        h5f = h5py.File(file_path, 'r')
        np_images = np.asarray(h5f.get("images"))
        np_labels = np.asarray(h5f.get("labels")).astype(int)
        return np_images, np_labels


def loader_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load', action='store_true'
    )
    parser.add_argument(
        '--pack', action='store_true'
    )
    parser.add_argument(
        '--size_scalability', type=int, default=None
    )
    parser.add_argument(
        '--no_gpu', action='store_true'
    )
    args = parser.parse_args()
    db_creds = DbCredentials(db_name=cats.db_name,
                             user=cats.user,
                             host=cats.host,
                             port=cats.port,
                             password=cats.password)
    return args, db_creds


if __name__ == "__main__":
    args, db_creds = loader_args()
    imagenet_loader = ImageNetLoader(
        db_creds, 16, size_scalability=args.size_scalability,
        no_gpu=args.no_gpu)

    name_list = ['imagenet_train_data', 'imagenet_valid_data']
    purpose_list = ['train', 'valid']
    if args.load:
        print("START LOADING")
        file_list_list = [get_all(
            cats.train_root, 'h5'), get_all(cats.valid_root, 'h5')]
        for name, file_list in zip(name_list, file_list_list):
            imagenet_loader.drop_table(name)
            imagenet_loader.load_many(file_list, name)
        print("END LOADING")
    if args.pack:
        print("START PACKING")
        imagenet_loader.create_binding()
        train_name, valid_name = name_list
        train_packed_name = imagenet_loader.get_pack_name(train_name)[0]
        for purpose in purpose_list:
            print("START PACKING: {}".format(purpose))
            imagenet_loader.pack(train_name, purpose,
                                 valid_name, train_packed_name)
            print("END PACKING: {}".format(purpose))
        print("END PACKING")
