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
from matplotlib import pyplot as plt
import pandas as pd
import datetime
import os
import numpy as np
import sys
import dill
import glob
import traceback
import psycopg2
import ast
from multiprocessing import Process
from multiprocessing import Manager

# sql = "select count(*) from table;"
# dat = pd.read_sql_query(sql, conn)
try:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('load_ext sql')
    ipython.magic('sql postgresql://gpadmin@localhost:5432/cerebro')
    conn = psycopg2.connect('postgresql://gpadmin@localhost:5432/cerebro')
except Exception as e:
    print(e)
    print("DB related functions not loaded")
sys.path.append('../cerebro_gpdb/')

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
CONTROL_IFACE_NAME = 'CONTROL_IFACE'
EXP_IFACE_NAME = 'EXPERIMENT_IFACE'
KEY = ['EPOCH', 'PARTITION']
DEBUG = False

# FIXME redundant re-def from cerebro_gpdb.utilis


def get_output_names_hyperopt(timestamp):
    output = 't_{timestamp}_udaf_hyperopt'.format(**locals())
    output_info = output + '_info'
    output_summary = output + '_summary'
    output_mst_table = output + '_mst_table'
    output_mst_table_summary = output + '_mst_table_summary'
    return output, output_info, \
        output_summary, output_mst_table, output_mst_table_summary


def parse_ctq_learning_curve(lines):
    filtered_lines = []
    for i, line in enumerate(lines):
        if 'ENDING BATCH:' in line:
            filtered_lines.append(lines[i - 1])

    batch_list = [ast.literal_eval(x) for x in filtered_lines]
    dict_list = []
    for i, batch in enumerate(batch_list):
        for mst_key, val_metrics in batch.items():
            batch_dict = {'batch': i}
            batch_dict['mst_key'] = mst_key
            batch_dict['validation_metrics'] = val_metrics
            dict_list.append(batch_dict)

    df_grand = pd.DataFrame(dict_list)
    return df_grand


def parse_madlib_line(line):
    _, info, timestamp = [x.strip() for x in line.split(': ')]
    timestamp = timestamp.split('(')[0].strip()
    time_obj = datetime.datetime.strptime(timestamp, TIME_FORMAT)
#     info = re.sub("\d", "", info)
    return info, time_obj, timestamp


def parse_ctq_line(line, return_timestamp=False):
    parsed = [x.strip() for x in line.split(': ')]
    info, timestamp = parsed[:-1], parsed[-1]
    time_obj = datetime.datetime.strptime(timestamp, TIME_FORMAT)
#     info = re.sub("\d", "", info)
    if not return_timestamp:
        return info, time_obj
    else:
        return info, time_obj, timestamp


def split_standalone_line(line):
    splitted = line.split(', ')
    splitted = [x.split(': ') for x in splitted]
    splitted = [x for x in splitted if len(x) == 2]
    info_dict = dict(splitted)
    return info_dict


def parse_standalone_line(line):
    info_dict = split_standalone_line(line)
    info = info_dict['EVENT']
    time_obj = datetime.datetime.fromtimestamp(int(info_dict['TIME']))
    timestamp = info_dict['TIME']
    return info, time_obj, timestamp


def ctq_parse_model_info_ordered(model_info_ordered):
    df_grand_list = []
    for k, v in model_info_ordered.items():
        for status_dict in v:
            status_dict['mst_str'] = k
            df_grand_list.append(status_dict)
    df_grand = pd.DataFrame(df_grand_list)
    return df_grand


def ctq_find(df_grand, mst_key=None, best=True, mode='metric'):
    df_aggregated = df_grand.groupby(
        ['epoch', 'model_key']).agg(
        {'loss_train': 'mean',
            'metric_train': 'mean',
            'loss_valid': 'mean',
            'metric_valid': 'mean'}).reset_index()

    epoch = 'epoch'
    mst_str = 'model_key'
    if mode == 'metric':
        valid_metric = 'metric_valid'
    else:
        valid_metric = 'loss_valid'
    max_epoch = df_aggregated[epoch].max()
    df_last_epoch = df_aggregated.loc[
        (df_aggregated[epoch] == max_epoch)].reset_index()
    if mode == 'metric':
        idx = df_last_epoch[[valid_metric]].idxmax()
    else:
        idx = df_last_epoch[[valid_metric]].idxmin()
    if best:
        best_mst_key = df_last_epoch.iloc[idx][mst_str].iloc[0]
        df_best = df_aggregated.loc[df_aggregated[mst_str] == best_mst_key]
    else:
        best_mst_key = mst_key
        df_best = df_aggregated.loc[df_aggregated[mst_str] == mst_key]
    df_plot = df_best
    x = np.array(df_plot[epoch])
    if mode == 'metric':
        y = 1 - np.array(df_plot[valid_metric])
    else:
        y = np.array(df_plot[valid_metric])
    return x, y, best_mst_key


def save_fig(fig, path):
    fig.savefig('{}.pdf'.format(path), transparent=True, bbox_inches='tight')
    fig.savefig('{}.eps'.format(path), transparent=True, bbox_inches='tight')
    fig.savefig('{}.png'.format(path), transparent=True,
                bbox_inches='tight', dpi=600)


def get_all_start_end(global_log_dir, time_format, method, find_first=True):
    start_end_dict = {}
    with open(global_log_dir) as f_in:
        lines = f_in.readlines()
        start_end = [None, None, None, None]
        for line in lines:
            line = line.rstrip()
            if not find_first and method not in line:
                continue
            if 'Start time' in line:
                time_string = ' '.join(line.split(' ')[-2:])
                start_end[0] = time_string
                start_end[2] = datetime.datetime.strptime(
                    time_string, time_format)
            elif 'End time' in line:
                time_string = ' '.join(line.split(' ')[-2:])
                start_end[1] = time_string
                start_end[3] = datetime.datetime.strptime(
                    time_string, time_format)
                start_end_dict[method] = start_end
            else:
                start_end = [None, None, None, None]
    return start_end_dict


def readlines(filedir):
    with open(filedir) as f_in:
        lines = f_in.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


class SystemLogAnalyzer(object):
    def __init__(
            self,
            logs_root,
            start_time_obj,
            end_time_obj,
            time_format, ltype='processor', verbose=DEBUG):
        self.logs_root = logs_root
        self.start_time_obj = start_time_obj
        self.end_time_obj = end_time_obj
        self.total_time = (self.end_time_obj -
                           self.start_time_obj).total_seconds()
        self.time_format = time_format
        self.ltype = ltype
        self.verbose = verbose

    def parse_processor(self, lines, i):
        timestamp = lines[i]
        utilizations = lines[i + 1]
        utilization = float(utilizations.split(',')[0][:-1])
        time_obj = datetime.datetime.strptime(
            timestamp, self.time_format)
        i += 2
        return i, time_obj, utilization

    def parse_memory(self, lines, i):
        timestamp = lines[i]
        utilizations = lines[i + 1]
        utilization = float(utilizations.split(',')[1][:-1])
        time_obj = datetime.datetime.strptime(
            timestamp, self.time_format)
        i += 2
        return i, time_obj, utilization

    def parse_disk(self, lines, i):
        timestamp = lines[i]
        infos = lines[i + 1]
        sig = 'MB_wrtn '
        pos = infos.find(sig)
        infos = infos[pos + len(sig):]
        infos = infos.split(' ')
        info_lists = []
        for info in [infos[:6], infos[6:]]:
            info_dict = {}
            for k, v in zip(
                    ['Device', 'tps', 'MB_read/s', 'MB_wrtn/s', 'MB_read', 'MB_wrtn'], info):
                info_dict[k] = float(v) if k != 'Device' else v
            info_lists.append(info_dict)
        time_obj = datetime.datetime.strptime(
            timestamp, self.time_format)
        i += 2
        return i, time_obj, info_lists

    def parse_network_line(self, line):
        utilization = line.split(', ')
        utilization = [x.split(': ') for x in utilization]
        line_dict = dict(utilization)
        for iface in [CONTROL_IFACE_NAME, EXP_IFACE_NAME]:
            if iface in line_dict:
                line_dict['IFACE'] = CONTROL_IFACE_NAME
                del(line_dict[iface])
        del(line_dict['IFACE'])
        return line_dict

    def parse_network(self, lines, i):
        timestamp = lines[i]
        control_utilizations = lines[i + 1]
        exp_utilizations = lines[i + 2]
        control_utilization = self.parse_network_line(control_utilizations)
        exp_utilization = self.parse_network_line(exp_utilizations)
        time_obj = datetime.datetime.strptime(timestamp, self.time_format)
        utilization = [control_utilization, exp_utilization]
        i += 3
        return i, time_obj, utilization

    def parse(self, lines, logfile_dir):
        i = 0
        worker_utilizations = []
        while i < len(lines) - 1:
            if i % (len(lines) // 5) == 0 and self.verbose:
                print("{}/{} lines parsed".format(i, len(lines)))
            try:
                if self.ltype == 'processor':
                    i, time_obj, utilization = \
                        self.parse_processor(
                            lines, i)
                elif self.ltype == 'memory':
                    i, time_obj, utilization = \
                        self.parse_memory(
                            lines, i)
                elif self.ltype == 'disk':
                    i, time_obj, utilization = \
                        self.parse_disk(
                            lines, i)
                else:
                    i, time_obj, utilization = self.parse_network(
                        lines, i)
                if self.start_time_obj <= time_obj and \
                        time_obj < self.end_time_obj:
                    worker_utilizations.append(utilization)
                if time_obj > self.end_time_obj and self.verbose:
                    print("Exceeded exp end time at {}, breaking".format(i))
                    break
            except Exception as e:
                if DEBUG:
                    traceback.print_exc()
                if self.verbose:
                    print(e)
                    print(
                        "Parsing line failure: {}, {}, {}, skipping".format(
                            logfile_dir, i, lines[i:i + 2]))
                i += 1

        return worker_utilizations

    def report_disk_fn(self, logfile_dir, res):
        worker_name = logfile_dir.split('_')[-1].split('.')[0]
        lines = readlines(os.path.join(self.logs_root, logfile_dir))
        worker_utilizations = self.parse(lines, logfile_dir)
        worker_utilizations = [y for x in worker_utilizations for y in x]
        df_worker = pd.DataFrame(worker_utilizations)
        df_worker['worker_name'] = worker_name
        res.append(df_worker)

    def par_apply(self, in_list, fn, args):
        processes = []
        manager = Manager()
        res = manager.list()
        for i, input_item in enumerate(in_list):
            proc = Process(target=fn,
                           args=[input_item, res] + args)
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()
        return list(res)

    def par_apply_dict(self, in_dict, fn, args):
        processes = []
        manager = Manager()
        res = manager.dict()
        for k, v in in_dict.items():
            proc = Process(target=fn,
                           args=[(k, v), res] + args)
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()
        return dict(res)

    def report_disk(self, separateio=False):
        df_grand = pd.DataFrame()
        res = self.par_apply(os.listdir(self.logs_root),
                             self.report_disk_fn, [])
        # for logfile_dir in os.listdir(self.logs_root):
        #     print(logfile_dir)
        #     worker_name = logfile_dir.split('_')[-1].split('.')[0]
        #     lines = readlines(os.path.join(self.logs_root, logfile_dir))
        #     worker_utilizations = self.parse(lines, logfile_dir)
        #     worker_utilizations = [y for x in worker_utilizations for y in x]
        #     df_worker = pd.DataFrame(worker_utilizations)
        #     df_worker['worker_name'] = worker_name
        for df_worker in res:
            df_grand = df_grand.append(df_worker, ignore_index=True)
        df_grand_workers = df_grand[df_grand['worker_name'] != 'master']

        df_workers_io = df_grand_workers.groupby(
            ['worker_name', 'Device'])[['MB_read', 'MB_wrtn']].agg(['min', 'max']).reset_index()

        reads = df_workers_io['MB_read', 'max'] - \
            df_workers_io['MB_read', 'min']
        wrtns = df_workers_io['MB_wrtn', 'max'] - \
            df_workers_io['MB_wrtn', 'min']
        df_workers_io['MB_io'] = reads + wrtns
        df_workers_io['MB_i'] = reads
        df_workers_io['MB_o'] = wrtns
        if not separateio:
            df_io = df_workers_io[['worker_name', 'MB_io']].groupby(
                'worker_name').agg(['sum']).reset_index()
            mean_io_MB = df_io['MB_io']['sum'].mean()
            sum_io_MB = df_io['MB_io']['sum'].sum()
            return df_grand, mean_io_MB, sum_io_MB
        else:
            reads_summary = df_workers_io[['worker_name', 'MB_i']].groupby(
                'worker_name').agg(['sum']).reset_index()
            wrtns_summary = df_workers_io[['worker_name', 'MB_o']].groupby(
                'worker_name').agg(['sum']).reset_index()
            print("Per worker reads (GB): {}, Total reads (GB): {}".format(
                reads_summary['MB_i'].mean() / 1024, reads_summary['MB_i'].sum() / 1024))
            print("Per worker writes (GB): {}, Total writes (GB): {}".format(
                wrtns_summary['MB_o'].mean() / 1024, wrtns_summary['MB_o'].sum() / 1024))
            return df_grand, reads_summary, wrtns_summary

    def report_utilization(self, *args, **kwargs):
        return self.report(*args, **kwargs)

    def report_utilization_fn(self, logfile_dir, res):
        if 'worker' not in logfile_dir:
            return
        print(logfile_dir)
        lines = readlines(os.path.join(self.logs_root, logfile_dir))
        worker_utilizations = self.parse(lines, logfile_dir)
        res.append(np.mean(worker_utilizations))

    def report(self):
        workers_utilizations = self.par_apply(os.listdir(
            self.logs_root), self.report_utilization_fn, [])
        # for logfile_dir in os.listdir(self.logs_root):

        #     if 'worker' not in logfile_dir:
        #         continue
        #     print(logfile_dir)
        #     lines = readlines(os.path.join(self.logs_root, logfile_dir))
        #     worker_utilizations = self.parse(lines, logfile_dir)

        #     workers_utilizations.append(np.mean(worker_utilizations))
        return np.mean(workers_utilizations)

    def report_network_fn(self, kv_pair, res):
        logfile_dir, _ = kv_pair
        worker_name = logfile_dir.split('_')[-1].split('.')[0]
        lines = readlines(os.path.join(self.logs_root, logfile_dir))
        worker_utilizations = self.parse(lines, logfile_dir)

        control_df = pd.DataFrame(
            [x[0] for x in worker_utilizations]).astype(float)
        exp_df = pd.DataFrame(
            [x[1] for x in worker_utilizations]).astype(float)
        control_df['iface'] = 'CONTROL'
        exp_df['iface'] = 'EXP'
        df_grand = exp_df.append(control_df, ignore_index=True)
        df_grand['rxtxkB/s'] = df_grand['rxkB/s'] + df_grand['txkB/s']
        res[worker_name] = df_grand

    def report_df(self):
        logfile_dir_dict = {
            logfile_dir: None for logfile_dir in os.listdir(self.logs_root)}
        workers_utilizations = self.par_apply_dict(
            logfile_dir_dict, self.report_network_fn, [])
        # for logfile_dir in os.listdir(self.logs_root):
        #     print(logfile_dir)
        #     worker_name = logfile_dir.split('_')[-1].split('.')[0]
        #     lines = readlines(os.path.join(self.logs_root, logfile_dir))
        #     worker_utilizations = self.parse(lines, logfile_dir)

        #     control_df = pd.DataFrame(
        #         [x[0] for x in worker_utilizations]).astype(float)
        #     exp_df = pd.DataFrame(
        #         [x[1] for x in worker_utilizations]).astype(float)
        #     control_df['iface'] = 'CONTROL'
        #     exp_df['iface'] = 'EXP'
        #     df_grand = exp_df.append(control_df, ignore_index=True)
        #     df_grand['rxtxkB/s'] = df_grand['rxkB/s'] + df_grand['txkB/s']
        #     workers_utilizations[worker_name] = df_grand
        stats_master = workers_utilizations['master'].groupby(
            'iface').agg(['mean', 'max'])
        df_concat = [v for k, v in workers_utilizations.items()
                     if k != 'master']
        num_workers = len(df_concat)
        df_workers = pd.concat(
            df_concat,
            ignore_index=True)
        self.df_workers = df_workers
        self.df_master = workers_utilizations['master']
        stats_workers = df_workers.groupby('iface').agg(['mean', 'max'])
        print(
            "master stats:{}\nworkers stats:{}".format(
                stats_master[['rxtxkB/s']].iloc[1],
                stats_workers[['rxtxkB/s']].iloc[1]))
        avg_network_kBs = stats_master.loc['EXP']['rxkB/s']['mean'] + \
            stats_workers.loc['EXP']['rxkB/s']['mean'] * num_workers
        total_network_kBs = avg_network_kBs * self.total_time

        # convert to GBs
        total_network_GBs = total_network_kBs / 1024 / 1024

        print("Total GB transmitted over network: {}".format(total_network_GBs))
        return stats_master, stats_workers


def parse_model_logs(model_logs):
    df_list = []
    for model_log_dir in model_logs:
        lines = readlines(model_log_dir)
        mst_key = int(os.path.basename(model_log_dir).split(os.extsep)[0])
        mst_str = None
        for line in lines:
            if 'MST' in line:
                mst_str = line.split(': ')[1]
            if 'UNIX TIME' in line:
                line_dict = parse_line(line)
                if 'VALID LOSS' in line_dict:
                    line_dict['mode'] = 'VALID'
                elif 'TRAIN LOSS' in line_dict:
                    line_dict['mode'] = 'TRAIN'
                line_dict['mst_key'] = mst_key
                line_dict['mst_str'] = mst_str
                df_list.append(line_dict)

    df_grand = pd.DataFrame(df_list)

    numeric_columns = [
        'EPOCH',
        'UNIX TIME',
        'TRAIN LOSS', 'ERROR', 'ERROR-1',
        'Model Building and Session Initialization Time',
        'Time',
        'Checkpoint Save Time',
        'VALID LOSS'
    ]

    df_grand[numeric_columns] = df_grand[numeric_columns].apply(pd.to_numeric)
    return df_grand


class PlotterBase(object):
    def __init__(self, xlabel=None, ylabel=None, title=None, set_ticks=False, figsize=(5, 5)):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.set_ticks = set_ticks

    def plot(self, x, *args, **kwargs):
        self.ax.plot(x, linewidth=2, *args, **kwargs)
        if self.set_ticks:
            self.ax.set_xticks(x)

    def conclude(self, loc=None):
        self.ax.legend(ncol=2, loc=loc)
        self.fig.tight_layout()

    def save(self, path):
        save_fig(self.fig, path)


# class PlotterE2E(PlotterBase):
#     def __init__(self, *args, **kwargs):

#         super(PlotterE2E, self).__init__(*args, **kwargs)
#         self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
#         self.ax.set_xlabel('Epoch')
#         self.ax.set_ylabel('Top-5 Validation Error')

#     def plot(self, x, y, label):
#         self.ax.plot(x, y, label=label)
#         self.ax.set_xticks(x)

#     def conclude(self):
#         self.fig.legend(ncol=2, loc='upper right')
#         self.fig.tight_layout()


def parse_line(line):
    line_frags = []
    for item in line.split(', '):
        line_frags.append(item.split(': '))
    line_dict = dict(line_frags)
    return line_dict


def read_ma_data(exp_name, models_list, timestamp, force=False):
    pickle_name = '{}_{}.pickle'.format(exp_name, timestamp)
    print("LOOKING FOR {}".format(pickle_name))
    if os.path.exists(pickle_name) and not force:
        df_grand = pd.read_pickle(pickle_name)
    else:
        # First time load from db
        df_grand = pd.DataFrame()
        for name in models_list:
            df = ipython.magic(
                '''sql SELECT *
                FROM {name} model 
                JOIN model_arch_library arch ON model.model_id = arch.model_id
                '''.format(**locals())).DataFrame()

            df_grand = df_grand.append(df, ignore_index=True)
        df_grand = df_grand.sort_values(
            by=['validation_metrics_final'], ascending=False)
        try:
            df_grand.columns = [
                'source_table', 'model',
                'dependent_varname', 'independent_varname',
                'model_arch_table', 'model_id', 'compile_params', 'fit_params',
                'num_iterations', 'validation_table',
                'metrics_compute_frequency', 'name',
                'description', 'model_type', 'model_size', 'start_training_time',
                'end_training_time', 'metrics_elapsed_time', 'madlib_version',
                'num_classes', 'class_values',
                'dependent_vartype', 'normalizing_const',
                'metrics_type', 'loss_type', 'training_metrics_final', 'training_loss_final',
                'training_metrics', 'training_loss', 'validation_metrics_final',
                'validation_loss_final', 'validation_metrics', 'validation_loss',
                'metrics_iters', 'model_id_dup',
                'model_arch', 'model_weights', 'model_name',
                'description', '__internal_madlib_id__'
            ]
        except Exception:
            df_grand.columns = [
                'source_table', 'model',
                'dependent_varname', 'independent_varname',
                'model_arch_table', 'model_id', 'compile_params', 'fit_params',
                'num_iterations', 'validation_table',
                'metrics_compute_frequency', 'name',
                'description', 'model_type', 'model_size', 'start_training_time',
                'end_training_time', 'metrics_elapsed_time', 'madlib_version',
                'num_classes', 'class_values',
                'dependent_vartype', 'normalizing_const',
                'metrics_type', 'training_metrics_final', 'training_loss_final',
                'training_metrics', 'training_loss', 'validation_metrics_final',
                'validation_loss_final', 'validation_metrics', 'validation_loss',
                'metrics_iters', 'model_id_dup',
                'model_arch', 'model_weights', 'model_name',
                'description', '__internal_madlib_id__'
            ]

        df_grand = df_grand.drop('model_id_dup', 1)
        df_grand.to_pickle(pickle_name)
    df_plot = df_grand[['validation_metrics',
                        'model_name', 'model_id', 'metrics_iters']]
    return df_grand, df_plot


def read_udaf_hyperopt(exp_name, timestamp, force=False):
    pickle_name_info = '{}_info_{}.pickle'.format(exp_name, timestamp)
    pickle_name_mst_table = '{}_mst_table_{}.pickle'.format(
        exp_name, timestamp)
    pickle_name_summary = '{}_summary_{}.pickle'.format(exp_name, timestamp)
    _, output_info, \
        output_summary, output_mst_table, _ = get_output_names_hyperopt(
            timestamp)
    local_paths = {
        'info': (pickle_name_info, output_info),
        'mst_table': (pickle_name_mst_table, output_mst_table),
        'summary': (pickle_name_summary, output_summary)
    }

    df_dict = {}
    for name, (path, table_name) in local_paths.items():
        if os.path.exists(path) and not force:
            print("Found local pickles for {}, reading".format(path))
            df = pd.read_pickle(path)
        else:
            sql = '''SELECT *
                    FROM {}
                '''.format(table_name)
            df = pd.read_sql_query(sql, conn)
            df.to_pickle(path)
        df_dict[name] = df
    return df_dict


def read_cerebro_data(exp_name, timestamp, force=False):
    pickle_name_summary = '{}_summary_{}.pickle'.format(exp_name, timestamp)
    pickle_name_info = '{}_info_{}.pickle'.format(exp_name, timestamp)
    if os.path.exists(pickle_name_summary) and not force:
        df_summary = pd.read_pickle(pickle_name_summary)
    else:
        # First time load from db
        df_summary = ipython.magic('''sql SELECT *
                FROM t_{timestamp}_mop_summary
            '''.format(**locals())).DataFrame()
        df_summary.to_pickle(pickle_name_summary)
    if os.path.exists(pickle_name_info) and not force:
        df_info = pd.read_pickle(pickle_name_info)
    else:
        info_query = '''sql SELECT *
            FROM t_{timestamp}_mop_info model JOIN
                model_arch_library arch ON model.model_id = arch.model_id
            '''.format(**locals())
        df_info = ipython.magic(info_query).DataFrame()
        print("Info query: {}".format(info_query))
        if df_info is None:
            raise ValueError(
                "Failed to retrieve from DB, query: {}".format(info_query))
        df_info = df_info.sort_values(
            by=['validation_metrics_final'],
            ascending=False).reset_index(drop=True)
        try:
            df_info.columns = [
                'mst_key', 'model_id', 'compile_params',
                'fit_params', 'model_type',
                'model_size', 'metrics_elapsed_time', 'metrics_type', 'loss_type',
                'training_metrics_final',
                'training_loss_final', 'training_metrics',
                'training_loss', 'validation_metrics_final',
                'validation_loss_final',
                'validation_metrics', 'validation_loss',
                'model_id_dup', 'model_arch',
                'model_weights', 'model_name',
                'description', '__internal_madlib_id__']
        except Exception:
            df_info.columns = [
                'mst_key', 'model_id', 'compile_params',
                'fit_params', 'model_type',
                'model_size', 'metrics_elapsed_time', 'metrics_type',
                'training_metrics_final',
                'training_loss_final', 'training_metrics',
                'training_loss', 'validation_metrics_final',
                'validation_loss_final',
                'validation_metrics', 'validation_loss',
                'model_id_dup', 'model_arch',
                'model_weights', 'model_name',
                'description', '__internal_madlib_id__']
        df_info = df_info.drop('model_id_dup', 1)
        df_info.to_pickle(pickle_name_info)
    df_plot = df_info[['validation_metrics', 'model_name', 'model_id']].copy()
    df_plot['metrics_iters'] = [df_summary['metrics_iters'][0]
                                for _ in range(len(df_plot))]
    return df_summary, df_info, df_plot


def cal_time_diff(grand_zero, time_obj):
    try:
        return (time_obj - grand_zero).total_seconds() / 3600.0
    except:
        return (time_obj - grand_zero) / 3600.0


class LogAnalyzer(object):
    def __init__(
            self,
            log_root,
            gpu_logs,
            cpu_logs,
            timestamp,
            method,
            exp_name,
            timestamp_db=None,
            num_models=None,
            network_logs=None,
            disk_logs=None,
            find_first=False,
            match_exp_name=False):
        self.log_root = log_root
        self.run_log_root = os.path.join(log_root, 'run_logs', timestamp)
        self.gpu_log_root = os.path.join(
            log_root, gpu_logs) if gpu_logs else None
        self.cpu_log_root = os.path.join(
            log_root, cpu_logs) if cpu_logs else None
        self.network_logs = network_logs
        if network_logs:
            self.network_log_root = os.path.join(log_root, network_logs)
        if disk_logs:
            self.disk_log_root = os.path.join(log_root, disk_logs)
        self.time_format = TIME_FORMAT
        self.global_log_dir = os.path.join(self.run_log_root,
                                           'global.log')
        self.exp_root = os.path.join(self.run_log_root, exp_name)
        if 'udaf-' in method:
            self.method = 'udaf'
        elif 'ctq-' in method:
            self.method = 'ctq'
        else:
            self.method = method

        self.match_name = method if not match_exp_name else exp_name
        self.start_end_dict = get_all_start_end(
            self.global_log_dir,
            self.time_format, self.match_name, find_first=find_first)
        self.timestamp_db = timestamp_db
        self.num_models = num_models

        self.exp_name = exp_name

    def find_best(self, df_grand, df_plot=None, top_5=True):
        y_col = 'ERROR' if top_5 else 'ERROR-1'
        if self.method == 'ctq':
            x, y, _ = ctq_find(df_grand, best=True)
        elif self.method in ['ma', 'udaf']:
            x = np.array(df_plot.iloc[0]['metrics_iters'])
            y = 1 - np.array(df_plot.iloc[0]['validation_metrics'])
        elif self.method in ['ddp', 'da-ddp']:
            df_aggregated = df_grand.groupby(
                ['epoch', 'model_index', 'mode']).agg(
                {'loss': 'mean',
                 'acc_top1': 'mean',
                 'acc_top5': 'mean'}).reset_index()
            df_best = df_aggregated[
                (df_aggregated['mode'] == 'valid') & (
                    df_aggregated['epoch'] == df_aggregated['epoch'].max())]
            best_model_row = df_best['acc_top5'].idxmax()

            best_model_index = df_best.loc[best_model_row]['model_index']

            df_best_model = df_aggregated[
                (df_aggregated['mode'] == 'valid') & (
                    df_aggregated['model_index'] == best_model_index)]

            x, y = df_best_model['epoch'].values, \
                df_best_model['acc_top5'].values
            x = 1 + x
            y = 1 - y
        else:
            df_aggregated = self.agg_df_grand_cerebro(df_grand)
            max_epoch = df_aggregated['EPOCH'].max()

            df_last_epoch = df_aggregated.loc[
                (df_aggregated['EPOCH'] == max_epoch) &
                (df_aggregated['mode'] == 'VALID')].reset_index()
            idx = df_last_epoch[[y_col]].idxmin()

            best_mst_key = df_last_epoch.iloc[idx]['mst_key'].iloc[0]

            df_best = df_aggregated.loc[df_aggregated['mst_key']
                                        == best_mst_key]
            df_plot = df_best.loc[df_best['mode']
                                  == 'VALID'][['EPOCH', y_col]]
            x = np.array(df_plot['EPOCH']) + 1
            y = np.array(df_plot[y_col])
        return x, y

    def agg_df_grand_cerebro(self, df_grand):

        df_aggregated = df_grand.groupby(
            ['EPOCH', 'mst_key', 'mode', 'mst_str']).agg(
            {
                'TRAIN LOSS': 'mean',
                'VALID LOSS': 'mean',
                'ERROR': 'mean',
                'ERROR-1': 'mean'}).reset_index()
        return df_aggregated

    def get_runtimes(self):
        start_time_obj = self.start_end_dict[self.match_name][2]
        end_time_obj = self.start_end_dict[self.match_name][3]
        total_time = (end_time_obj - start_time_obj).total_seconds() / 3600
        return start_time_obj, end_time_obj, total_time

    def get_df_grand(self):
        df_plot = None
        if 'ma' in self.method:
            models_list = ['t_{}_m_{}_summary'.format(
                self.timestamp_db, i) for i in range(1, self.num_models + 1)]
            df_grand, df_plot = read_ma_data(
                self.exp_name, models_list, self.timestamp_db)
        elif 'udaf' in self.method:
            df_summary, df_info, df_plot = read_cerebro_data(
                self.exp_name, self.timestamp_db)
            df_grand = df_info
        elif 'ctq' in self.method:
            model_info_filepath = os.path.join(
                self.run_log_root, self.exp_name, 'models_info.pkl')
            with open(model_info_filepath, "rb") as f:
                df_grand_dict = dill.load(f)
            df_grand = ctq_parse_model_info_ordered(df_grand_dict)
        elif self.method in ['da', 'filesystem', 'da-hyperopt', 'cerebro_standalone-hyperopt']:
            model_logs = glob.glob(os.path.join(self.exp_root, '*.log'))
            model_logs = [
                x for x in model_logs if
                'scheduler' not in x and 'client' not in x]
            df_grand = parse_model_logs(model_logs)
        elif self.method in ['ddp', 'da-ddp']:
            model_logs = glob.glob(os.path.join(self.exp_root, '*.pkl'))
            comm_name = 'all_logs_rank'
            index_name = 'index'
            model_logs = [x for x in model_logs if comm_name in x]
            grand_df_list = []
            for filename in model_logs:
                with open(filename, "rb") as f:
                    log = dill.load(f)
                    log_name = os.path.basename(filename).split('.')[0]
                    log_name = log_name[len(comm_name):]
                    rank, index_str = log_name.split('_')
                    index = index_str[len(index_name):]
                    rank = int(rank)
                    index = int(index)
                    for e, dd in log.items():
                        for pur, ddd in dd.items():
                            status_dict = {}

                            status_dict['loss'] = ddd['_'.join([pur, 'loss'])]
                            status_dict['acc_top1'] = ddd['_'.join(
                                [pur, 'acc_top1'])]
                            status_dict['acc_top5'] = ddd['_'.join(
                                [pur, 'acc_top5'])]
                            status_dict['mode'] = pur
                            status_dict['epoch'] = e
                            status_dict['rank'] = rank
                            status_dict['model_index'] = index
                            grand_df_list.append(status_dict)

            df_grand = pd.DataFrame(grand_df_list)
        return df_grand, df_plot

    def get_learning_curve(self, top_5=True):
        self.df_grand, self.df_plot = self.get_df_grand()
        x, y = self.find_best(self.df_grand, self.df_plot, top_5)
        return self.df_grand, x, y

    def report_hyperopt(self, shift=0, concurrency=8, epochs=10):
        grand_zero, _, _ = self.shift_time(shift)
        if self.exp_name == 'spark-hyperopt':
            all_logs = glob.glob(os.path.join(self.exp_root, 'batch_size*'))
            df_subs = []
            xaxis = []
            for log_filename in all_logs:
                basename = os.path.basename(log_filename)
                mst_key = basename.split('.log')[0]
                lines = readlines(log_filename)
                start_line, history_line = [
                    x for x in lines if 'MST: ' in x or 'HISTORY: ' in x]
                _, start_time_obj = parse_ctq_line(start_line)
                infos, end_time_obj = parse_ctq_line(history_line)
                infos = ': '.join(infos[1:])
                info_dict = ast.literal_eval(infos)
                y = info_dict['val_top_k_categorical_accuracy']
                start_time = (start_time_obj -
                              grand_zero).total_seconds() / 3600.0
                end_time = (end_time_obj - grand_zero).total_seconds() / 3600.0
                x = np.linspace(start_time, end_time, epochs + 1)[1:]
                df_sub = pd.DataFrame(
                    [{'mst_key': mst_key, 'validation_metrics': y}])
                df_subs.append(df_sub)
                xaxis.append(x)
            return xaxis, df_subs
        if self.exp_name == 'udaf-hyperopt':
            self.df_dict = read_udaf_hyperopt(self.exp_name, self.timestamp_db)
            lines = readlines(os.path.join(self.exp_root, 'client.log'))
            parsed_lines = [parse_madlib_line(
                x) for x in lines if 'End MOP training epoch' in x and 'subepoch' not in x]
            df_info = self.df_dict['info'].sort_values('mst_key')
        elif self.exp_name == 'ctq-hyperopt':
            lines = readlines(os.path.join(self.exp_root, 'client.log'))
            df_info = parse_ctq_learning_curve(lines)
            filtered_lines = [line for line in lines if 'LEFT JOBS: 0' in line]
            parsed_lines = [parse_ctq_line(x) for x in filtered_lines]

        elif self.exp_name == 'cerebro_spark-hyperopt':
            lines = readlines(os.path.join(self.exp_root, 'client.log'))
            df_grand_dict = []
            for line in lines:
                if 'CEREBRO => Time' in line and 'Model: ' in line and 'Epoch: ' in line and 'val_top_k_categorical_accuracy: ' in line:
                    splitted = split_standalone_line(line)
                    for k, v in splitted.items():
                        if k == 'CEREBRO => Time':
                            splitted[k] = datetime.datetime.strptime(
                                v, TIME_FORMAT)
                        elif k == 'Model':
                            splitted[k] = int(v.split('_')[1])
                        elif k == 'Epoch':
                            splitted[k] = int(v)
                        elif k != 'Model':
                            splitted[k] = float(v)
                    df_grand_dict.append(splitted)

            df_grand = pd.DataFrame(df_grand_dict)

            df_info = df_grand.sort_values(['Model', 'Epoch']).groupby(
                ['Model']).agg({'val_top_k_categorical_accuracy': list}).reset_index()

            df_info.rename(columns={'val_top_k_categorical_accuracy': 'validation_metrics',
                                    'Model': 'mst_idx'}, inplace=True)
            max_config = len(df_info)
            df_grand['Batch'] = df_grand['Model'] // concurrency

            df_ends = df_grand.groupby(['Batch', 'Epoch']).agg(
                {'CEREBRO => Time': 'max'}).reset_index()

            parsed_lines = df_ends[[
                'Epoch', 'CEREBRO => Time']].values.tolist()
        else:
            # standalone methods
            lines = readlines(os.path.join(self.exp_root, 'client.log'))
            filtered_lines = [x for x in lines if 'EPOCH' in x]
            df_dict_list = []
            df_dict_list = [split_standalone_line(x) for x in filtered_lines]
            df_time = pd.DataFrame(df_dict_list)
            df_time_casted = df_time[[
                x for x in df_time.columns if x != 'EVENT']].astype(int)
            df_time_casted['EVENT'] = df_time['EVENT']
            df_time = df_time_casted
            df_time = df_time.sort_values('TIME')
            df_ends = df_time.groupby(['ITERATION', 'EPOCH']).agg(
                {'TIME': 'max'}).reset_index()
            parsed_lines = df_ends[['ITERATION', 'TIME']].values.tolist()

            df_grand, df_plot = self.get_df_grand()
            df_aggregated = self.agg_df_grand_cerebro(df_grand)
            df_valid = df_aggregated[df_aggregated['mode'] == 'VALID']
            df_valid['ERROR'] = 1 - df_valid['ERROR']
            df_info = df_valid.sort_values(['mst_key', 'EPOCH']).groupby(
                ['mst_key', 'mst_str']).agg({'ERROR': list}).reset_index()
            df_info.rename(columns={'ERROR': 'validation_metrics',
                                    'mst_key': 'mst_idx', 'mst_str': 'mst_key'}, inplace=True)
            grand_zero = shift
        xaxis = []
        i = 0
        while i < len(parsed_lines):
            bag = parsed_lines[i:i+epochs]
            xaxis.append(
                [cal_time_diff(grand_zero, x[1]) for x in bag])
            i += epochs
        df_subs = []
        start = 0
        while start < len(df_info):
            df_subs.append(df_info.iloc[start:start+concurrency])
            start += concurrency
        return xaxis, df_subs

    def report(self, learning_curve=True, shift=0):
        self.df_grand = None
        x = None
        y = None

        total_time = self.report_runtimes()
        avg_gpu_utilization = self.report_gpu_utilization(shift)

        if learning_curve:
            self.df_grand, x, y = self.get_learning_curve()
        return self.method, \
            total_time, avg_gpu_utilization, x, y, self.df_grand

    def shift_time(self, shift):
        start_time_obj, end_time_obj, total_time = self.get_runtimes()
        print("Original time: {} to {}".format(start_time_obj, end_time_obj))
        start_time_obj += datetime.timedelta(seconds=shift)
        print("Shifted time: {} to {}".format(start_time_obj, end_time_obj))
        return start_time_obj, end_time_obj, total_time

    def report_network(self, shift=0):
        start_time_obj, end_time_obj, total_time = self.shift_time(shift)
        # method = self.method
        # start_time_obj = self.start_end_dict[method][2]
        # end_time_obj = self.start_end_dict[method][3]
        sys_la = SystemLogAnalyzer(
            self.network_log_root, start_time_obj,
            end_time_obj, self.time_format,
            ltype='network')
        self.network_stats_master, \
            self.network_stats_workers = sys_la.report_df()

        return self.network_stats_master, self.network_stats_workers

    def report_disk(self, shift=0, separateio=False):
        start_time_obj, end_time_obj, total_time = self.shift_time(shift)
        sys_la = SystemLogAnalyzer(
            self.disk_log_root, start_time_obj,
            end_time_obj, self.time_format,
            ltype='disk')
        df_grand = sys_la.report_disk(separateio=separateio)
        return df_grand

    def report_runtimes(self):
        start_time_obj, end_time_obj, total_time = self.get_runtimes()
        print("Method: {}".format(self.method))
        print("Total time: {} hrs".format(total_time))
        return total_time

    def report_processor_memory_utilization(
            self, shift, name='GPU', ltype='processor'):
        start_time_obj, end_time_obj, total_time = self.shift_time(shift)
        if name == 'GPU':
            log_root = self.gpu_log_root
        else:
            log_root = self.cpu_log_root
        sys_la = SystemLogAnalyzer(log_root, start_time_obj,
                                   end_time_obj, self.time_format, ltype=ltype)
        avg_utilization = sys_la.report()
        print("Average {} utilization: {} %".format(
            name + ltype, avg_utilization))
        return avg_utilization

    def report_gpu_utilization(self, shift=0):
        return self.report_processor_memory_utilization(
            shift, 'GPU', 'processor')

    def report_cpu_utilization(self, shift=0):
        return self.report_processor_memory_utilization(
            shift, 'CPU', 'processor')

    def report_gpu_memory_utilization(self, shift=0):
        return self.report_processor_memory_utilization(shift, 'GPU', 'memory')

    def report_cpu_memory_utilization(self, shift=0):
        return self.report_processor_memory_utilization(shift, 'CPU', 'memory')
