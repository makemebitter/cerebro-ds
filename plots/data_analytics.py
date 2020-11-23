from matplotlib import pyplot as plt
import pandas as pd
import datetime
import os
import numpy as np
import sys
import dill
import glob
import traceback

try:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('load_ext sql')
    ipython.magic('sql postgresql://gpadmin@localhost:5432/cerebro')
except Exception as e:
    print(e)
    print("DB related functions not loaded")
sys.path.append('../cerebro_gpdb/')

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
CONTROL_IFACE_NAME = 'CONTROL_IFACE'
EXP_IFACE_NAME = 'EXPERIMENT_IFACE'
KEY = ['EPOCH', 'PARTITION']
DEBUG = False


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
            if i % (len(lines)//5) == 0 and self.verbose:
                print("{}/{} lines parsed".format(i, len(lines)))
            try:
                if self.ltype == 'processor':
                    i, time_obj, utilization = \
                        self.parse_processor(
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
                            logfile_dir, i, lines[i:i+2]))
                i += 1

        return worker_utilizations

    def report(self):
        workers_utilizations = []
        for logfile_dir in os.listdir(self.logs_root):
            
            if 'worker' not in logfile_dir:
                continue
            print(logfile_dir)
            lines = readlines(os.path.join(self.logs_root, logfile_dir))
            worker_utilizations = self.parse(lines, logfile_dir)

            workers_utilizations.append(np.mean(worker_utilizations))
        return np.mean(workers_utilizations)

    def report_df(self):
        workers_utilizations = {}
        for logfile_dir in os.listdir(self.logs_root):
            print(logfile_dir)
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
            workers_utilizations[worker_name] = df_grand
        stats_master = workers_utilizations['master'].groupby(
            'iface').agg(['mean', 'max'])
        df_workers = pd.concat(
            [v for k, v in workers_utilizations.items() if k != 'master'],
            ignore_index=True)
        self.df_workers = df_workers
        self.df_master = workers_utilizations['master']
        stats_workers = df_workers.groupby('iface').agg(['mean', 'max'])
        print(
            "master stats:{}\nworkers stats:{}".format(
                stats_master[['rxtxkB/s']].iloc[1],
                stats_workers[['rxtxkB/s']].iloc[1]))
        return stats_master, stats_workers


def parse_model_logs(model_logs):
    df_list = []
    for model_log_dir in model_logs:
        lines = readlines(model_log_dir)
        mst_key = os.path.basename(model_log_dir).split(os.extsep)[0]
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


def read_ma_data(exp_name, models_list, force=False):
    pickle_name = '{}.pickle'.format(exp_name)
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


def read_cerebro_data(exp_name, timestamp, force=False):
    pickle_name_summary = '{}_summary.pickle'.format(exp_name)
    pickle_name_info = '{}_info.pickle'.format(exp_name)
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
        df_info = ipython.magic(
            '''sql SELECT *
            FROM t_{timestamp}_mop_info model JOIN
                model_arch_library arch ON model.model_id = arch.model_id
            '''.format(**locals())).DataFrame()
        df_info = df_info.sort_values(
            by=['validation_metrics_final'],
            ascending=False).reset_index(drop=True)
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
            find_first=False):
        self.log_root = log_root
        self.run_log_root = os.path.join(log_root, 'run_logs', timestamp)
        self.gpu_log_root = os.path.join(log_root, gpu_logs) if gpu_logs else None
        self.cpu_log_root = os.path.join(log_root, cpu_logs) if cpu_logs else None
        self.network_logs = network_logs
        if network_logs:
            self.network_log_root = os.path.join(log_root, network_logs)
        self.time_format = TIME_FORMAT
        self.global_log_dir = os.path.join(self.run_log_root,
                                           'global.log')
        self.exp_root = os.path.join(self.run_log_root, exp_name)
        self.method = method
        self.start_end_dict = get_all_start_end(
            self.global_log_dir,
            self.time_format, self.method, find_first=find_first)
        self.timestamp_db = timestamp_db
        self.num_models = num_models

        self.exp_name = exp_name

    def find_best(self, df_grand, df_plot=None, top_5=True):
        y_col = 'ERROR' if top_5 else 'ERROR-1'
        if self.method == 'ctq':
            df_aggregated = df_grand.groupby(
                ['epoch', 'model_key']).agg(
                {'loss_train': 'mean',
                 'metric_train': 'mean',
                 'loss_valid': 'mean',
                 'metric_valid': 'mean'}).reset_index()

            epoch = 'epoch'
            mst_str = 'model_key'
            valid_metric = 'metric_valid'

            max_epoch = df_aggregated[epoch].max()

            df_last_epoch = df_aggregated.loc[
                (df_aggregated[epoch] == max_epoch)].reset_index()

            idx = df_last_epoch[[valid_metric]].idxmax()

            best_mst_key = df_last_epoch.iloc[idx][mst_str].iloc[0]

            df_best = df_aggregated.loc[df_aggregated[mst_str] == best_mst_key]

            df_plot = df_best
            x = np.array(df_plot[epoch])
            y = 1 - np.array(df_plot[valid_metric])
        elif self.method in ['ma', 'udaf']:
            x = np.array(df_plot.iloc[0]['metrics_iters'])
            y = 1 - np.array(df_plot.iloc[0]['validation_metrics'])
        else:
            df_aggregated = df_grand.groupby(
                ['EPOCH', 'mst_key', 'mode']).agg(
                {
                    'TRAIN LOSS': 'mean',
                    'VALID LOSS': 'mean',
                    'ERROR': 'mean',
                    'ERROR-1': 'mean'}).reset_index()
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

    def get_runtimes(self):
        method = self.method
        start_time_obj = self.start_end_dict[method][2]
        end_time_obj = self.start_end_dict[method][3]
        total_time = (end_time_obj - start_time_obj).total_seconds() / 3600
        return start_time_obj, end_time_obj, total_time

    def get_df_grand(self):
        df_plot = None
        if 'ma' in self.method:
            models_list = ['t_{}_m_{}_summary'.format(
                self.timestamp_db, i) for i in range(1, self.num_models + 1)]
            df_grand, df_plot = read_ma_data(self.exp_name, models_list)
        elif 'udaf' in self.method:
            df_summary, df_info, df_plot = read_cerebro_data(
                self.exp_name, self.timestamp_db)
            df_grand = df_info
        elif 'ctq' in self.method:
            model_info_filepath = os.path.join(
                self.run_log_root, self.exp_name, 'models_info.pkl')
            with open(model_info_filepath, "rb") as f:
                df_grand_dict = dill.load(f)
            df_grand_list = []
            for k, v in df_grand_dict.items():
                for status_dict in v:
                    status_dict['mst_str'] = k
                    df_grand_list.append(status_dict)
            df_grand = pd.DataFrame(df_grand_list)
        elif any([x in self.method for x in ['da', 'filesystem']]):
            model_logs = glob.glob(os.path.join(self.exp_root, '*.log'))
            model_logs = [
                x for x in model_logs if
                'scheduler' not in x and 'client' not in x]
            df_grand = parse_model_logs(model_logs)
        return df_grand, df_plot

    def get_learning_curve(self, top_5=True):
        self.df_grand, self.df_plot = self.get_df_grand()
        x, y = self.find_best(self.df_grand, self.df_plot, top_5)
        return self.df_grand, x, y

    def report(self, learning_curve=True, shift=0):
        self.df_grand = None
        x = None
        y = None

        total_time = self.report_runtimes()
        avg_gpu_utilization = self.report_gpu_tuilization(shift)

        if learning_curve:
            self.df_grand, x, y = self.get_learning_curve()
        return self.method, \
            total_time, avg_gpu_utilization, x, y, self.df_grand

    def report_network(self):
        method = self.method
        start_time_obj = self.start_end_dict[method][2]
        end_time_obj = self.start_end_dict[method][3]
        sys_la = SystemLogAnalyzer(
            self.network_log_root, start_time_obj,
            end_time_obj, self.time_format,
            ltype='network')
        self.network_stats_master, \
            self.network_stats_workers = sys_la.report_df()
        return self.network_stats_master, self.network_stats_workers

    def report_runtimes(self):
        start_time_obj, end_time_obj, total_time = self.get_runtimes()
        print("Method: {}".format(self.method))
        print("Total time: {} hrs".format(total_time))
        return total_time
    def report_processor_utilization(self, shift=0, name='GPU'):
        start_time_obj, end_time_obj, total_time = self.get_runtimes()
        print("Original time: {} to {}".format(start_time_obj, end_time_obj))
        start_time_obj += datetime.timedelta(seconds=shift)
        print("Shifted time: {} to {}".format(start_time_obj, end_time_obj))
        if name == 'GPU':
            log_root = self.gpu_log_root
        else:
            log_root = self.cpu_log_root
        sys_la = SystemLogAnalyzer(log_root, start_time_obj,
                                   end_time_obj, self.time_format)
        avg_utilization = sys_la.report()
        print("Average {} utilization: {} %".format(name, avg_utilization))
        return avg_utilization
    
    def report_gpu_tuilization(self, shift=0):
        avg_gpu_utilization = self.report_processor_utilization(shift, 'GPU')
        return avg_gpu_utilization
    
    def report_cpu_tuilization(self, shift=0):
        avg_cpu_utilization = self.report_processor_utilization(shift, 'CPU')
        return avg_cpu_utilization
