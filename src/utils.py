import datetime
import psycopg2
from madlib_keras_wrapper import serialize_nd_weights
import select
import random
import time
import json
import keras
import os
import pandas as pd
import sys
import numpy as np
from keras import backend as K
import tensorflow as tf
from imagenetcat import SEED

MINIBATCH_OUTPUT_DEPENDENT_COLNAME_DL = "dependent_var"
MINIBATCH_OUTPUT_INDEPENDENT_COLNAME_DL = "independent_var"
DISTRIBUTION_KEY_COLNAME = "__dist_key__"
mb_dep_var_col = MINIBATCH_OUTPUT_DEPENDENT_COLNAME_DL
mb_indep_var_col = MINIBATCH_OUTPUT_INDEPENDENT_COLNAME_DL
dep_shape_col = 'dependent_var_shape'
ind_shape_col = 'independent_var_shape'
dist_key_col = DISTRIBUTION_KEY_COLNAME
GP_SEGMENT_ID_COLNAME = "gp_segment_id"
CUDA_VISIBLE_DEVICES_KEY = 'CUDA_VISIBLE_DEVICES'


def tstamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def logs(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("{}: {}".format(message, timestamp))
    sys.stdout.flush()



def set_seed(SEED=SEED):
    # Seed value
    # Apparently you may use different seed values at each stage

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

    os.environ['PYTHONHASHSEED'] = str(SEED)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value

    random.seed(SEED)

    # 3. Set the `numpy` pseudo-random generator at a fixed value

    np.random.seed(SEED)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value

    tf.random.set_random_seed(SEED)
    # for later versions:
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    # for later versions:
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)


class ModelArchSchema:
    col_names = (
        'model_id', 'model_arch',
        'model_weights', 'name', 'description',
        '__internal_madlib_id__')
    col_types = ('SERIAL PRIMARY KEY', 'JSON', 'bytea', 'TEXT', 'TEXT', 'TEXT')
    (MODEL_ID, MODEL_ARCH, MODEL_WEIGHTS, NAME, DESCRIPTION,
     __INTERNAL_MADLIB_ID__) = col_names


def set_cuda_env(value):
    """
    :param value: -1 to disable gpu
    :return:
    """
    os.environ[CUDA_VISIBLE_DEVICES_KEY] = value


def get_device_name_and_set_cuda_env(gpu_count, seg):
    if gpu_count > 0:
        device_name = '/gpu:0'
        cuda_visible_dev = str(seg % gpu_count)
        set_cuda_env(cuda_visible_dev)
    else:  # cpu only
        device_name = '/cpu:0'
        set_cuda_env('-1')
    return device_name


def get_initial_weights(
        model_table, model_arch, serialized_weights, warm_start=None,
        use_gpus=None, accessible_gpus_for_seg=None, mst_filter=''):
    _ = get_device_name_and_set_cuda_env(0, None)
    if warm_start:
        raise NotImplementedError("Warm start not implemented")
    else:
        if not serialized_weights:
            model = keras.models.model_from_json(model_arch)
            serialized_weights = serialize_nd_weights(
                model.get_weights())
    return serialized_weights


def add_postfix(quoted_string, postfix):
    """ Append a string to the end of the table name.
    If input table name is quoted by double quotes, make sure the postfix is
    inside of the double quotes.

    Arguments:
        @param quoted_string: str. A string representing a database 
            quoted string
        @param postfix: str. A string to add as a suffix to quoted_string.
                            ** This is assumed to not contain any quotes **
    """
    quoted_string = quoted_string.strip()
    if quoted_string.startswith('"') and quoted_string.endswith('"'):
        output_str = quoted_string[:-1] + postfix + '"'
    else:
        output_str = quoted_string + postfix
    return output_str


def unique_string(desp='', **kwargs):
    """
    Generate random remporary names for temp table and other names.
    It has a SQL interface so both SQL and Python functions can call it.
    """
    r1 = random.randint(1, 100000000)
    r2 = int(time.time())
    r3 = int(time.time()) % random.randint(1, 100000000)
    u_string = "__madlib_temp_" + desp + \
        str(r1) + "_" + str(r2) + "_" + str(r3) + "__"
    return u_string


def wait(conn):
    while 1:
        state = conn.poll()
        while conn.notices:
            print(conn.notices.pop(0))
        if state == psycopg2.extensions.POLL_OK:
            break
        elif state == psycopg2.extensions.POLL_WRITE:
            select.select([], [conn.fileno()], [])
        elif state == psycopg2.extensions.POLL_READ:
            select.select([conn.fileno()], [], [])
        else:
            raise psycopg2.OperationalError("poll() returned %s" % state)


class cats(object):
    db_name = 'cerebro'
    user = 'gpadmin'
    host = 'localhost'
    port = '5432'
    password = ''

class cats_imagenet(cats):
    train_root = '/mnt/nfs/imagenet/train'
    valid_root = '/mnt/nfs/imagenet/valid'

class cats_criteo(cats):
    train_root = '/mnt/nfs/hdd/criteo/npy/train'
    valid_root = '/mnt/nfs/hdd/criteo/npy/valid'

class DBBase(object):
    def __init__(self, db_creds=cats, a_sync=0):
        self.connection = psycopg2.connect(user=db_creds.user,
                                           password=db_creds.password,
                                           host=db_creds.host,
                                           port=db_creds.port,
                                           database=db_creds.db_name,
                                           async_=a_sync)
        if not a_sync:
            self.connection.autocommit = True
            self.cursor = self.connection.cursor()
        else:
            wait(self.connection)
            self.cursor = self.connection.cursor()

    def if_exists_table(self, name):
        res = None
        try:
            self.cursor.execute("SELECT '{}'::regclass".format(name))
            res = self.cursor.fetchone()
        except Exception:
            pass
        return res is not None

    def drop_table(self, name):
        self.cursor.execute("DROP TABLE IF EXISTS {}".format(name))

    def get_image_count_per_seg_for_minibatched_data_from_db(self, table_name):
        mb_dep_var_col = MINIBATCH_OUTPUT_DEPENDENT_COLNAME_DL
        shape_col = add_postfix(mb_dep_var_col, "_shape")
        query_string = """ SELECT {0}, sum({1}[1]) AS images_per_seg
                FROM {2}
                GROUP BY {0}
            """.format(DISTRIBUTION_KEY_COLNAME, shape_col, table_name)
        self.cursor.execute(query_string)
        wait(self.connection)
        images_per_seg = self.cursor.fetchall()
        seg_ids = [int(each_segment[0])
                   for each_segment in images_per_seg]
        images_per_seg = [int(each_segment[1])
                          for each_segment in images_per_seg]
        return seg_ids, images_per_seg

    def get_segments_per_host(self):
        self.cursor.execute("""
            SELECT count(*) from gp_segment_configuration
            WHERE role = 'p' and content != -1
            GROUP BY hostname
            LIMIT 1
            """)
        wait(self.connection)
        count = self.cursor.fetchone()[0]
        # in case some weird gpdb configuration happens, always returns
        # primary segment number >= 1
        return max(1, count)

    def get_seg_number(self):
        """ Find out how many primary segments(not include master segment) exist
            in the distribution. Might be useful for partitioning data.
        """
        self.cursor.execute("""
            SELECT count(*) from gp_segment_configuration
            WHERE role = 'p' and content != -1
            """)
        wait(self.connection)
        count = self.cursor.fetchone()[0]
        # in case some weird gpdb configuration happens, always returns
        # primary segment number >= 1
        return max(1, count)

    def get_accessible_gpus_for_seg(self, schema_madlib, segments_per_host, module_name):
        gpu_info_table = unique_string(desp='gpu_info')
        gpu_table_query = """
            SELECT {schema_madlib}.gpu_configuration('{gpu_info_table}')
        """.format(**locals())
        self.cursor.execute(gpu_table_query)
        wait(self.connection)
        gpu_query = """
            SELECT hostname, count(*) AS count FROM {gpu_info_table} GROUP BY hostname
            """.format(**locals())
        self.cursor.execute(gpu_query)
        wait(self.connection)
        gpu_query_result = self.cursor.fetchall()
        self.cursor.execute("DROP TABLE IF EXISTS {0}".format(gpu_info_table))
        wait(self.connection)
        if not gpu_query_result:
            raise Exception(
                "{0} error: No GPUs configured on hosts.".format(module_name))
        host_dict = {}
        for i in gpu_query_result:
            host_dict[i[0]] = int(i[1])

        seg_query = """
            SELECT hostname, content AS segment_id
            FROM gp_segment_configuration
            WHERE content != -1 AND role = 'p'
        """
        self.cursor.execute(seg_query)
        wait(self.connection)
        seg_query_result = self.cursor.fetchall()
        accessible_gpus_for_seg = [0] * len(seg_query_result)
        warning_flag = True
        for i in seg_query_result:
            if i[0] in host_dict.keys():
                accessible_gpus_for_seg[i[1]] = host_dict[i[0]]
            if 0 < accessible_gpus_for_seg[i[1]] < segments_per_host and warning_flag:
                print(
                    'The number of GPUs per segment host is less than the number of '
                    'segments per segment host. When different segments share the '
                    'same GPU, this may fail in some scenarios. The current '
                    'recommended configuration is to have 1 GPU available per segment.')
                warning_flag = False
        return accessible_gpus_for_seg

    def get_model_arch_weights(self, model_arch_table, model_id):
        # assume validation is already called
        model_arch_query = "SELECT {0}, {1} FROM {2} WHERE {3} = {4}".format(
            ModelArchSchema.MODEL_ARCH, ModelArchSchema.MODEL_WEIGHTS,
            model_arch_table, ModelArchSchema.MODEL_ID,
            model_id)
        self.cursor.execute(model_arch_query)
        wait(self.connection)
        model_arch_result = self.cursor.fetchall()
        if not model_arch_result:
            raise Exception(
                "no model arch found in table {0} with id {1}".format(
                    model_arch_table, model_id))

        model_arch_result = model_arch_result[0]

        model_arch = model_arch_result[0]
        model_arch = json.dumps(model_arch)
        model_weights = model_arch_result[1]

        return model_arch, model_weights


class DBConnect(object):
    def __init__(self, db_creds):
        self.adb = DBBase(db_creds, 1)
        self.aconnection = self.adb.connection
        self.acursor = self.adb.cursor
        self.db = DBBase(db_creds, 0)
        self.connection = self.db.connection
        self.cursor = self.db.cursor
        self.cursor.execute("SET OPTIMIZER TO OFF")
        self.acursor.execute("SET OPTIMIZER TO OFF")
        wait(self.aconnection)
        wait(self.connection)
        self.mb_dep_var_col = MINIBATCH_OUTPUT_DEPENDENT_COLNAME_DL
        self.mb_indep_var_col = MINIBATCH_OUTPUT_INDEPENDENT_COLNAME_DL
        self.dep_shape_col = add_postfix(
            MINIBATCH_OUTPUT_DEPENDENT_COLNAME_DL, "_shape")
        self.ind_shape_col = add_postfix(
            MINIBATCH_OUTPUT_INDEPENDENT_COLNAME_DL, "_shape")
        self.dist_key_col = DISTRIBUTION_KEY_COLNAME
        self.gp_segment_id_col = GP_SEGMENT_ID_COLNAME
        self.segments_per_host = self.db.get_segments_per_host()

    def pd_query(self, query_string, schema):
        self.cursor.execute(query_string)
        res = self.cursor.fetchall()
        df = pd.DataFrame(res, columns=schema)
        return df
