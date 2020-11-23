"""Summary

Deleted Attributes:
    BOUNDARIES_BUCKET (TYPE): Description
    data_catalog (TYPE): Description
    data_root (TYPE): Description
    data_set_name (str): Description
    INDEX_CAT_FEATURES (int): Description
    log_file (TYPE): Description
    NB_BUCKETS (int): Description
    NB_INPUT_FEATURES (TYPE): Description
    NB_OF_HASHES_CAT (TYPE): Description
    train_availability (TYPE): Description
    train_partitions (TYPE): Description
    valid_availability (TYPE): Description
    valid_partitions (TYPE): Description
    VOCABULARY_SIZE (int): Description
    workers (TYPE): Description
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(os.getcwd())), "../../", "code"))
sys.path.append(os.path.dirname(
    os.path.realpath(__file__)) + "/../../code")
import argparse
from shutil import copyfile
import time
import pandas as pd
import numpy as np
import mmh3
from catalog import get_workers, get_data_catalog
import xmlrpclib
import base64
import dill
from utils import uuid
import tensorflow as tf
from pympler import asizeof
from hurry.filesize import size
from np_to_tfrecords import np_to_tfrecords



dill.settings["recurse"] = True
NUM_PARALLEL_CALLS = 40

VOCABULARY_SIZE = 39
INDEX_CAT_FEATURES = 13
NB_OF_HASHES_CAT = 2 ** 8
NB_BUCKETS = 50

BOUNDARIES_BUCKET = [1.5 ** j - 0.51 for j in range(NB_BUCKETS)]
NB_INPUT_FEATURES = (INDEX_CAT_FEATURES * NB_BUCKETS) + \
    ((VOCABULARY_SIZE - INDEX_CAT_FEATURES) * NB_OF_HASHES_CAT)


def _get_index_bucket(feature_value):
    """Summary

    Args:
        feature_value (TYPE): Description

    Returns:
        TYPE: Description
    """
    for index, boundary_value in enumerate(BOUNDARIES_BUCKET):
        if feature_value < boundary_value:
            return index
    return index


def _get_next_input(input):
    """Summary

    Args:
        input (TYPE): Description

    Returns:
        TYPE: Description
    """
    data = np.zeros(NB_INPUT_FEATURES)
    features = input[1:]

    if len(features) != VOCABULARY_SIZE:
        return data, 0

    # Bucketing continuous features
    for f_index in range(0, INDEX_CAT_FEATURES):
        if features[f_index]:
            bucket_index = _get_index_bucket(int(features[f_index]))
            bucket_number_index = f_index * NB_BUCKETS
            bucket_index_offset = bucket_index + bucket_number_index
            data[bucket_index_offset] = 1

    # Bucketing categorical features
    offset = INDEX_CAT_FEATURES * NB_BUCKETS
    for f_index in range(INDEX_CAT_FEATURES, VOCABULARY_SIZE):
        if features[f_index]:
            hash_index = mmh3.hash(features[f_index]) % NB_OF_HASHES_CAT
            hash_number_index = (
                f_index - INDEX_CAT_FEATURES) * NB_OF_HASHES_CAT + offset
            hash_index_offset = hash_index + hash_number_index
            data[hash_index_offset] = 1

    data = data.astype(np.float32)
    nz_idx = np.nonzero(data)
    return nz_idx[0].astype(np.int64), np.take(data, nz_idx[0]), [input[0]]


def _extract_features(data):
    """Summary

    Args:
        data (TYPE): Description

    Returns:
        TYPE: Description
    """
    return [_get_next_input(data[i]) for i in range(data.shape[0])]


def _gen_dataset(data, batch_size):
    """Summary

    Args:
        data (TYPE): Description
        batch_size (TYPE): Description

    Returns:
        TYPE: Description

    Deleted Parameters:
        mst (TYPE): Description
    """
    def _data_generator(data):
        """Summary

        Args:
            data (TYPE): Description

        Yields:
            TYPE: Description
        """
        for i in range(len(data)):
            yield data[i]

    def _sparse_to_dense_tensor(x_id, x_val, y):
        """Summary

        Args:
            x_id (TYPE): Description
            x_val (TYPE): Description
            y (TYPE): Description

        Returns:
            TYPE: Description
        """
        return tf.sparse_tensor_to_dense(tf.SparseTensor(indices=x_id, values=x_val, dense_shape=[NB_INPUT_FEATURES])), y
    dataset = tf.data.Dataset.from_generator(lambda: _data_generator(data), output_types=(tf.int64, tf.float32, tf.float32),
                                             output_shapes=((None), (None), (1)))\
        .map(lambda x_id, x_val, y: _sparse_to_dense_tensor(x_id, x_val, y), num_parallel_calls=NUM_PARALLEL_CALLS)\
        .batch(batch_size)\
        .prefetch(3)
    return dataset


def preprocess_fn(dummy_var,
                  data_partitions,
                  batch_size,
                  frac=None,
                  random_state=None,
                  nfs_dirname=None):
    """Summary

    Args:
        dummy_var (TYPE): Description
        data_partitions (TYPE): Description
        batch_size (TYPE): Description
        frac (None, optional): Description
        random_state (None, optional): Description
    """
    for file_path in data_partitions:
        basename = os.path.basename(file_path)
        dirname = os.path.dirname(file_path)
        filename, file_extension = os.path.splitext(basename)
        out_file_local = os.path.join(dirname, '.'.join([filename, 'npy']))
        out_file_local_compressed = os.path.join(
            dirname, '.'.join([filename + '_compressed', 'npz']))
        out_file_local_tfrecords = os.path.join(dirname, filename)
        out_file_nfs_tfrecords = os.path.join(nfs_dirname, filename)
        print("OUTFILE: {}".format(out_file_local_tfrecords))
        print("LOADING: {}".format(file_path))
        print("LOADING DATA IN PANDAS")
        df = pd.read_csv(file_path, sep="\t", header=None)
        print("FINISH LOADING DATA IN PANDAS")
        print("FILLING NAs")
        df.fillna(0, inplace=True)
        print("SAMPLING")
        df = df.sample(frac=frac, random_state=random_state)
        data = df.values
        print("DF TO LIST")
        data = [_get_next_input(data[i]) for i in range(data.shape[0])]
        del df
        print("PRE-ALLOCATING")
        dataset_mat = dataset_mat = np.zeros(
            (len(data), NB_INPUT_FEATURES + 1), dtype=np.float32)
        begin_time = time.time()
        count = 0
        data_iterator = _gen_dataset(
            data, batch_size).make_one_shot_iterator()
        x, y = data_iterator.get_next()
        with tf.Session() as sess:
            print("STARTING TENSORFLOW OPERATIONS")
            while True:
                try:
                    x_val, y_val = sess.run([x, y])
                    dataset_batch = np.hstack((x_val, y_val))
                    dataset_mat[count:count +
                                dataset_batch.shape[0]] = dataset_batch
                    count += len(dataset_batch)
                    elapsed_time = time.time() - begin_time
                    RPS = count / elapsed_time
                    curr_RSS = size(asizeof.asizeof(dataset_mat))
                    if count % (batch_size * 10) == 0 \
                            or count < batch_size \
                            or count == len(data):
                        print('RPS: {}, RSS: {}, Progress: {}%, ETA: {}sec'
                              .format(
                                  RPS,
                                  curr_RSS,
                                  count / len(data) * 100,
                                  (len(data) - count) / RPS),
                              file=sys.stderr)
                except tf.errors.OutOfRangeError:
                    break
        try:
            assert dataset_mat.shape[0] == len(data)
            print('Assertion passed!', file=sys.stderr)
        except AssertionError:
            print('AssertionError', file=sys.stderr)
            print(len(dataset_mat), len(data), file=sys.stderr)
        # free memory
        del data
        # local copy
        np.save(out_file_local, dataset_mat)
        np.savez_compressed(out_file_local_compressed, dataset_mat=dataset_mat)
        np_to_tfrecords(dataset_mat[:, :-1], dataset_mat[:, -1,
                                                         np.newaxis], out_file_local_tfrecords, verbose=True)
        print("COPYING TO NFS")
        copyfile(out_file_local_tfrecords + '.tfrecords',
                 out_file_nfs_tfrecords + '.tfrecords')
        del dataset_mat


def preprocess(preprocess_fn,
               log_file,
               batch_size,
               frac=None,
               random_state=None, nfs_dirname=None):
    """Summary

    Args:
        preprocess_fn (TYPE): Description
        log_file (TYPE): Description
        batch_size (TYPE): Description
        frac (None, optional): Description
        random_state (None, optional): Description

    Raises:
        Exception: Description
    """
    begin_time = time.time()
    preprocess_fn_string = base64.b64encode(
        dill.dumps(preprocess_fn, byref=False))
    for worker in workers:
        worker.initialize_worker()
    exec_ids = []
    for worker_id, worker in enumerate(workers):
        data_partitions = []
        for availability, partitions in zip([train_availability, valid_availability],
                                            [train_partitions, valid_partitions]):
            for i, available in enumerate(availability[worker_id]):
                if available:
                    data_partitions.append((data_root + "/" + partitions[i]))

        exec_id = uuid()
        params = [data_partitions, batch_size, frac, random_state, nfs_dirname]

        status = dill.loads(base64.b64decode(
            worker.execute(exec_id, preprocess_fn_string, params)))

        if status != "LAUNCHED":
            raise Exception("Remote job launch failed. Reason: " + status)

        exec_ids.append((exec_id, worker_id))

    # wait for everything to finish
    while len(exec_ids) > 0:
        for exec_id, worker_id in exec_ids:
            worker = workers[worker_id]
            status = dill.loads(base64.b64decode(worker.status(exec_id)))

            if status["status"] == "FAILED":
                print(status)
                raise Exception("Remote job execution failed")
            elif status["status"] == "INVALID ID":
                raise Exception("Invalid Id")
            elif status["status"] == "COMPLETED":
                exec_ids.remove((exec_id, worker_id))
                message = "TIME: %d, EVENT: PREPROCESSING_COMPLETED, WORKER: %d\n" % (
                    time.time() - begin_time, worker_id)
                log_file.write(message)
                print(message[:-1])
                log_file.flush()
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', type=str, default=''
    )
    parser.add_argument(
        '--log_root', type=str, default='/data/nfs/data_share/criteo/run_logs'
    )
    parser.add_argument(
        '--nfs_root', type=str, default='/data/nfs/data_share/criteo/tfrecords'
    )
    args = parser.parse_args()
    data_set_name = 'criteo'
    data_catalog = get_data_catalog()
    workers = get_workers()
    workers = [xmlrpclib.ServerProxy(w) for w in workers]
    if not args.data_root:
        data_root = data_catalog[data_set_name]["data_root"]
    else:
        data_root = args.data_root
    train_partitions = data_catalog[data_set_name]["train"]
    train_availability = data_catalog[data_set_name]["train_availability"]
    valid_partitions = data_catalog[data_set_name]["valid"]
    valid_availability = data_catalog[data_set_name]["valid_availability"]
    log_file = open(
        os.path.join(args.log_root, "preprocess.log"), "w+")
    preprocess(preprocess_fn, log_file, 512, frac=0.025, random_state=42,
               nfs_dirname=args.nfs_root)
