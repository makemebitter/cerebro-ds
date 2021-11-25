import os
import tensorflow as tf
from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator
from cerebro.storage import HDFSStore
from cerebro.storage import LocalStore as NFSStore
from cerebro.tune import GridSearch, hp_choice, TPESearch
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
import dill
from cerebro_gpdb.imagenetcat import spark_imagenet_cat_nfs
from cerebro_gpdb.imagenetcat import param_grid
from cerebro_gpdb.criteocat import param_grid_criteo
from cerebro_gpdb.criteocat import spark_criteo_cat_nfs
from cerebro_gpdb.imagenetcat import param_grid_hyperopt
from cerebro_gpdb.imagenetcat import TOP_1
from cerebro_gpdb.imagenetcat import TOP_5
from cerebro_gpdb.single_node_helper import create_optimizer_from_mst
from cerebro_gpdb.in_rdbms_helper import create_model_from_mst
from cerebro_gpdb.utils import LOG_KEYS
from cerebro_gpdb.utils import logsc
from cerebro_gpdb.utils import logs
import numpy as np

os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3.7'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3.7'
os.environ['ARROW_LIBHDFS_DIR'] = '/local/hadoop/lib/native'
NFS_PATH = '/mnt/nfs/hdd/cerebro_spark_tmp'
NFS = 'nfs'
HDFS_PATH = 'hdfs://master:9000/cerebro_spark_tmp'
HDFS = 'hdfs'
DEBUG = False
if DEBUG:
    START = 0
    END = 1
else:
    START = 0
    END = 8


class CerebroSparkImageNetBase(object):
    def __init__(
            self,
            num_workers,
            store='hdfs',
            store_path='hdfs://master:9000/tmp',
            args=None,
            param_grid=param_grid):

        cores = '1' if args.spark_data_prepared else '32'
        memory = '150G' if args.spark_data_prepared else '450G'
        self.spark = SparkSession \
            .builder \
            .master("spark://master:7077") \
            .config('spark.python.worker.memory', memory)\
            .config("spark.executor.memory", memory)\
            .config("spark.memory.fraction", "0.8")\
            .config('spark.default.parallelism', str(args.size))\
            .config("spark.executor.cores", cores)\
            .config("spark.executorEnv.HADOOP_HOME", "/local/hadoop")\
            .config("spark.executorEnv.ARROW_LIBHDFS_DIR", "/local/hadoop/lib/native/")\
            .appName("CerebroSparkImageNet") \
            .getOrCreate()
        self.num_workers = num_workers
        self.backend = SparkBackend(spark_context=self.spark.sparkContext,
                                    num_workers=num_workers,
                                    disk_cache_size_gb=100,
                                    start_timeout=60000000,
                                    verbose=2,
                                    nics=['enp94s0f0'],
                                    data_readers_pool_type='process',
                                    num_data_readers=10)
        if store == 'hdfs':
            self.store = HDFSStore(store_path)
        elif store == 'nfs':
            self.store = NFSStore(store_path)
        self.args = args
        self.param_grid = param_grid

    def estimator_gen_fn(self, mst):
        optimizer = create_optimizer_from_mst(mst)
        model = create_model_from_mst(mst, module='tf.keras')
        loss = tf.keras.losses.CategoricalCrossentropy()
        keras_estimator = SparkEstimator(model=model,
                                         optimizer=optimizer,
                                         loss=loss,
                                         metrics=[TOP_1, TOP_5],
                                         batch_size=mst['batch_size'])
        return keras_estimator

    def load(self, spark_imagenet_cat):
        for i, path in enumerate(spark_imagenet_cat.valid_list[START:END]):
            if i == 0:
                df = self.load_one(path, 1)
            else:
                df_new = self.load_one(path, 1)
                df = df.union(df_new)
        for i, path in enumerate(spark_imagenet_cat.train_list[START:END]):
            df_new = self.load_one(path, 0)
            df = df.union(df_new)
        logs("Original num partitions: {}".format(df.rdd.getNumPartitions()))
        # df = df.repartition(self.num_workers * 32)
        arr_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
        df = df.select(
            arr_to_vector_udf(df["labels"]).alias("labels"),
            arr_to_vector_udf(df["features"]).alias("features"),
            df["validation"])
        return df

    def load_one(self, path, validation):
        df = self.spark.read.format("parquet").load(path)
        df = df.withColumn("features", df['features'].cast("array<float>")).\
            withColumn("labels", df['labels'].cast("array<float>"))
        df = df.withColumn("validation", F.lit(validation))
        return df

    def prepare_data(self, spark_imagenet_cat, prepared=False):
        if prepared:
            logs("Skip data loading")
            logs("Skip data preparing")
        else:
            with logsc(LOG_KEYS.DATA_LOADING):
                logs("Data loading")
                df = self.load(spark_imagenet_cat)
                logs("Data ETL")
                self.backend.prepare_data(
                    self.store, df, validation='validation')

    def prepare_search(self, epoch):
        raise NotImplementedError

    def run(self, spark_imagenet_cat, epoch=10, prepared=False):
        self.prepare_data(spark_imagenet_cat, prepared=prepared)
        if self.args.spark_data_prepared:
            self.prepare_search(epoch=epoch)
            if not self.args.hyperopt:
                logs("param_maps:{}".format(self.search.estimator_param_maps))
            else:
                logs("hyperopt_search_space:{}".format(
                    self.search.hyperopt_search_space))
            with logsc(LOG_KEYS.TRAINING):
                model = self.search.fit_on_prepared_data()
            return model


class CerebroSparkImageNetGridSearch(CerebroSparkImageNetBase):
    def prepare_search(self, epoch=10):
        self.search_space = {key: hp_choice(value)
                             for key, value in self.param_grid.items()}
        self.search = GridSearch(self.backend,
                                 self.store,
                                 self.estimator_gen_fn,
                                 self.search_space,
                                 epoch,
                                 evaluation_metric='loss',
                                 feature_columns=['features'],
                                 label_columns=['labels'],
                                 verbose=2)


class CerebroSparkImageNetHyperopt(CerebroSparkImageNetBase):
    def prepare_search(self, epoch=10):
        from cerebro_gpdb.hyperopt_helper import init_hyperopt
        self.search_space, _ = init_hyperopt(
            self.param_grid, search_space_only=True, cerebro=True)
        self.search = TPESearch(self.backend,
                                self.store,
                                self.estimator_gen_fn,
                                self.search_space,
                                num_models=self.args.max_num_config,
                                num_epochs=epoch,
                                evaluation_metric='loss',
                                feature_columns=['features'],
                                label_columns=['labels'],
                                verbose=2)


def schedule_grid_search(
        args
):
    schedule_cerebro_spark(args, 'grid')


def schedule_hyperopt(
        args
):
    schedule_cerebro_spark(args, 'hyperopt')


def schedule_cerebro_spark(args, method):
    if method == 'grid':
        Runner = CerebroSparkImageNetGridSearch
    elif method == 'hyperopt':
        Runner = CerebroSparkImageNetHyperopt
    if args.criteo:
        cat = spark_criteo_cat_nfs
        grid = param_grid_criteo
    else:
        cat = spark_imagenet_cat_nfs
        if args.hyperopt:
            grid = param_grid_hyperopt
        else:
            grid = param_grid

    logs("START RUNNING")
    if args.hdfs:
        store = HDFS
        store_path = HDFS_PATH
    else:
        store = NFS
        store_path = NFS_PATH
    logs((store, store_path))
    runner = Runner(
        args.size, store=store, store_path=store_path, args=args, param_grid=grid)
    model = runner.run(cat,
                       args.num_epochs, args.spark_data_prepared)
    print("best_model_history:{}".format(model.get_best_model_history()))
    print("all_model_history:{}".format(model.get_all_model_history()))
    if not args.hyperopt:
        with open(os.path.join(args.logs_root, 'model_params.dill'), "wb") as f:
            dill.dump(runner.search.estimator_param_maps, f)
    with open(os.path.join(args.logs_root, 'history.dill'), "wb") as f:
        dill.dump(model.get_all_model_history(), f)
    logs("END RUNNING")
