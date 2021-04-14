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
from cerebro_gpdb.pathmagic import *  # noqa
from cerebro_gpdb.utils import DBConnect
import os
import dill
# from cerebro.code.catalog import get_data_catalog
from cerebro_gpdb.pg_page_reader import table_page_read
from cerebro_gpdb.pg_page_reader import toast_page_read
from cerebro_gpdb.utils import logs
import pandas as pd
import re
import numpy as np
SYS_CAT_PATH = '/mnt/nfs/sys_cat.dill'


def input_fn(
    file_path,
    overwrite_table_page_path=None,
    overwrite_toast_page_path=None
):
    file_path_splited = os.path.split(file_path)
    mode = file_path_splited[-1]
    logs("CURRENT MODE: {}".format(mode))
    root_dir = file_path_splited[0]
    with open(SYS_CAT_PATH, "rb") as f:
        sys_cats = dill.load(f)
    df_shape = sys_cats['shape']
    sys_cats = sys_cats[mode]

    gp_segment_id = int(re.search("gpseg(\d+)", file_path).group(1))
    sys_cat = sys_cats.loc[sys_cats['gp_segment_id'] == gp_segment_id].iloc[0]

    table_page_path = os.path.join(root_dir, str(int(sys_cat['relfilenode'])))
    toast_page_path = os.path.join(
        root_dir, str(int(sys_cat['toast_relfilenode'])))
    table_name = sys_cat['relname']
    if overwrite_table_page_path:
        table_page_path = overwrite_table_page_path
    if overwrite_toast_page_path:
        toast_page_path = overwrite_toast_page_path

    df_data, df_toast = table_page_read(table_page_path)
    df_actual_data = toast_page_read(
        toast_page_path, df_toast, df_shape, table_name)
    return df_actual_data


class DirectAccessClient(DBConnect):
    def __init__(self, db_creds, db_name, train_name,
                 valid_name, size=8):
        super(DirectAccessClient, self).__init__(db_creds)
        self.db_name = db_name
        self.train_name = train_name
        self.valid_name = valid_name
        if 'imagenet' in self.train_name:
            self.name_like = 'imagenet'
        elif 'criteo' in self.train_name:
            self.name_like = 'criteo'
        self.size = size
        self.actual_size = 8 if size == 1 else size
        with open("../gp_configs/gphost_list_{}".format(size), 'r') as f:
            host_list = f.readlines()
            self.host_list = sorted([x.rstrip() for x in host_list])
        self.workers = [
            "http://{}:8000".format(x) for x in self.host_list
        ]
        self.segment_ids = [
            re.search("worker(\d+)", x).group(1) for x in self.host_list]
        print(self.workers)

    def get_df_att_user(self, relid):
        query_string = """ SELECT * FROM pg_catalog.pg_attribute where attrelid={}
                    """.format(relid)
        schema = [
            'attrelid', 'attname', 'atttypid', 'attstattarget', 'attlen',
            'attnum', 'attndims', 'attcacheoff', 'atttypmod', 'attbyval',
            'attstorage', 'attalign', 'attnotnull', 'atthasdef',
            'attisdropped', 'attislocal', 'attinhcount'
        ]
        df_att_user = self.pd_query(query_string, schema)
        return df_att_user

    def get_df_pagefiles(self):
        query_string = """
            select a.gp_segment_id, a.oid, a.relname, a.relfilenode,b.oid,
            b.relname, b.relfilenode, b.reltoastidxid from
            gp_dist_random('pg_class') a 
            LEFT OUTER JOIN gp_dist_random('pg_class') b 
            ON (a.reltoastrelid = b.oid and a.gp_segment_id = b.gp_segment_id)
            where a.relname like '%{}%';
            """.format(self.name_like)
        schema = [
            'gp_segment_id', 'oid', 'relname', 'relfilenode', 'toast_oid',
            'toast_relname', 'toast_relfilenode', 'reltoastidxid'
        ]
        df_pagefiles = self.pd_query(query_string, schema)
        return df_pagefiles

    def get_var_shape(self, table_name):
        query_string = """
        select __dist_key__, buffer_id, 
        independent_var_shape, dependent_var_shape 
        from {}""".format(table_name)
        schema = [
            '__dist_key__',
            'buffer_id',
            'independent_var_shape',
            'dependent_var_shape'
        ]
        df_var_shape = self.pd_query(query_string, schema)
        df_var_shape['table_name'] = table_name
        return df_var_shape

    def get_df_workers(self):
        query_string = """SELECT dbid,content,hostname,address  FROM 
        pg_catalog.gp_segment_configuration"""
        schema = ['dbid', 'content', 'hostname', 'address']
        df_pagefiles = self.pd_query(query_string, schema)
        return df_pagefiles

    def get_df_dboid(self):
        query_string = """SELECT oid,datname  FROM pg_catalog.pg_database"""
        schema = ['oid', 'datname']
        df_dboid = self.pd_query(query_string, schema)
        return df_dboid

    def get_df_shape(self, valid_name, train_name):
        df_valid_shape = self.get_var_shape(valid_name)
        df_train_shape = self.get_var_shape(train_name)
        df_shape = pd.concat([df_valid_shape, df_train_shape], axis=0)
        return df_shape

    def get_workers(self):
        return self.workers

    def cat_factory(self):
        avalibility = np.eye(self.size, dtype=int).tolist()
        if self.size == 1 or self.size == 8:
            data_root = '/mnt/gpdata'
        else:
            data_root = '/mnt/gpdata_{}'.format(self.size)
        cat = {
            'data_root': data_root,
            'train': ['gpseg{}/base'.format(x) for x in self.segment_ids],
            'train_availability': avalibility,
            'valid': ['gpseg{}/base'.format(x) for x in self.segment_ids],
            'valid_availability': avalibility
        }
        return cat

    def generate_cats(self):
        df_dboid = self.get_df_dboid()
        dboid = df_dboid.loc[df_dboid['datname']
                             == self.db_name]['oid'].iloc[0]
        data_cat = self.cat_factory()
        data_cat['train'] = [
            'gpseg{}/base/{}/train'.format(i, dboid) for i in self.segment_ids]
        data_cat['valid'] = [
            'gpseg{}/base/{}/valid'.format(i, dboid) for i in self.segment_ids]
        df_pagefiles = self.get_df_pagefiles()
        sys_cats = {}
        for mode, relname in zip(
                ['train', 'valid'], [self.train_name, self.valid_name]):
            rows = df_pagefiles.loc[df_pagefiles['relname'] == relname]
            sys_cats[mode] = rows
        df_shape = self.get_df_shape(self.valid_name, self.train_name)
        sys_cats['shape'] = df_shape
        with open(SYS_CAT_PATH, "wb") as f:
            dill.dump(sys_cats, f)
        return data_cat, sys_cats
