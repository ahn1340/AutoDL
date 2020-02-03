import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'ingestion'))
sys.path.append(os.path.join(os.getcwd(), 'scoring'))

# unused #
import signal
from contextlib import contextmanager
import math
# unused #
import multiprocessing
import torch
import random
import traceback
import numpy as np
import time
from sklearn.metrics import auc
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB
from dataset import AutoSpeechDataset
from score import get_solution, accuracy, is_multiclass, autodl_auc
from model import Model
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
import logging
import pickle
import librosa


LOGGER = logging.getLogger(__name__)

######################## Helper Functions ###########################
def extract_mfcc(data,sr=16000):
    results = []
    for d in data:
        r = librosa.feature.mfcc(d,sr=16000,n_mfcc=13)
        r = r.transpose()
        results.append(r)
    return results


def pad_seq(data,pad_len):
    return sequence.pad_sequences(data,maxlen=pad_len,dtype='float32',padding='pre')


def get_data_loader(train_X, train_y, batch_size=64):
    """Return train dataset. Test loader? Not needed as evaluation is done on the whole dataset? """
    train_X = torch.Tensor(train_X)
    train_y = torch.Tensor(train_y)
    train = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    return train_loader
######################## Helper Functions ###########################


###################### Timer for budget management #######################
class TimeoutException(Exception):
    pass

class Timer:
    def __init__(self):
        self.duration = 0
        self.total = None
        self.remain = None
        self.exec = None

    def set(self, time_budget):
        self.total = time_budget
        self.remain = time_budget
        self.exec = 0

    @contextmanager
    def time_limit(self, pname):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(math.ceil(self.remain)))
        start_time = time.time()

        try:
            yield
        finally:
            exec_time = time.time() - start_time
            signal.alarm(0)
            self.exec += exec_time
            self.duration += exec_time
            self.remain = self.total - self.exec

        if self.remain <= 0:
            raise TimeoutException("Timed out!")
###################### Timer for budget management #######################


def get_configspace(model):
    '''
    We define different configuration space for each model, as they have different hyperparameters
    '''
    cs = CS.ConfigurationSpace()
    # Model-independent hyperparamters
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='batch_size', choices = [16,32,64,128]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-3, upper=1e-1, log=True))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='optimizer', choices=['Adam', 'SGD']))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='weight_decay', lower=1e-2, upper=0.2, log=True))

    if model == 'LSTM':
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='last_n_outputs', lower=1, upper=50, log=False))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='num_layer', lower=1, upper=5, log=False))
        cs.add_hyperparameter((CSH.UniformIntegerHyperparameter(name='hidden_dim', lower=10, upper=100)))

    elif model == 'ESN':
        last_n_outputs = CSH.UniformIntegerHyperparameter(name='last_n_outputs', lower=1, upper=50, log=False)
        hidden_dim = CSH.UniformIntegerHyperparameter(name='hidden_dim', lower=10, upper=400)
        sparsity = CSH.UniformFloatHyperparameter(name='sparsity', lower=0.5, upper=0.99, log=False)
        leak_rate = CSH.UniformFloatHyperparameter(name='leak_rate', lower=0.5, upper=1, log=False)

        cs.add_hyperparameters([last_n_outputs,
                                hidden_dim,
                                sparsity,
                                leak_rate,
                                ])
    elif model == 'FCN':
        num_layer = CSH.UniformIntegerHyperparameter(name='num_layers', lower=1, upper=5, log=False)

        num_filter_1 = CSH.UniformIntegerHyperparameter(name='num_filters_1', lower=3, upper=128, log=True)
        num_filter_2 = CSH.UniformIntegerHyperparameter(name='num_filters_2', lower=3, upper=128, log=True)
        num_filter_3 = CSH.UniformIntegerHyperparameter(name='num_filters_3', lower=3, upper=128, log=True)
        num_filter_4 = CSH.UniformIntegerHyperparameter(name='num_filters_4', lower=3, upper=128, log=True)
        num_filter_5 = CSH.UniformIntegerHyperparameter(name='num_filters_5', lower=3, upper=128, log=True)

        kernel_size_1 = CSH.CategoricalHyperparameter(name='kernel_size_1', choices=[3,5,7])
        kernel_size_2 = CSH.CategoricalHyperparameter(name='kernel_size_2', choices=[3,5,7])
        kernel_size_3 = CSH.CategoricalHyperparameter(name='kernel_size_3', choices=[3,5,7])
        kernel_size_4 = CSH.CategoricalHyperparameter(name='kernel_size_4', choices=[3,5,7])
        kernel_size_5 = CSH.CategoricalHyperparameter(name='kernel_size_5', choices=[3,5,7])

        filter_cond_1 = CS.GreaterThanCondition(num_filter_2, num_layer, 1)
        filter_cond_2 = CS.GreaterThanCondition(num_filter_3, num_layer, 2)
        filter_cond_3 = CS.GreaterThanCondition(num_filter_4, num_layer, 3)
        filter_cond_4 = CS.GreaterThanCondition(num_filter_5, num_layer, 4)

        kernel_cond_1 = CS.GreaterThanCondition(kernel_size_2, num_layer, 1)
        kernel_cond_2 = CS.GreaterThanCondition(kernel_size_3, num_layer, 2)
        kernel_cond_3 = CS.GreaterThanCondition(kernel_size_4, num_layer, 3)
        kernel_cond_4 = CS.GreaterThanCondition(kernel_size_5, num_layer, 4)

        cs.add_hyperparameters([num_layer,
                                num_filter_1,
                                num_filter_2,
                                num_filter_3,
                                num_filter_4,
                                num_filter_5,
                                kernel_size_1,
                                kernel_size_2,
                                kernel_size_3,
                                kernel_size_4,
                                kernel_size_5,
                                ])

        cs.add_conditions([filter_cond_1,
                           filter_cond_2,
                           filter_cond_3,
                           filter_cond_4,
                           kernel_cond_1,
                           kernel_cond_2,
                           kernel_cond_3,
                           kernel_cond_4,
                           ])


    elif model == 'CNN_1D':
        num_layer = CSH.UniformIntegerHyperparameter(name='num_layers', lower=1, upper=5, log=False)

        num_filter_1 = CSH.UniformIntegerHyperparameter(name='num_filters_1', lower=3, upper=128, log=True)
        num_filter_2 = CSH.UniformIntegerHyperparameter(name='num_filters_2', lower=3, upper=128, log=True)
        num_filter_3 = CSH.UniformIntegerHyperparameter(name='num_filters_3', lower=3, upper=128, log=True)
        num_filter_4 = CSH.UniformIntegerHyperparameter(name='num_filters_4', lower=3, upper=128, log=True)
        num_filter_5 = CSH.UniformIntegerHyperparameter(name='num_filters_5', lower=3, upper=128, log=True)

        kernel_size_1 = CSH.CategoricalHyperparameter(name='kernel_size_1', choices=[3,5,7])
        kernel_size_2 = CSH.CategoricalHyperparameter(name='kernel_size_2', choices=[3,5,7])
        kernel_size_3 = CSH.CategoricalHyperparameter(name='kernel_size_3', choices=[3,5,7])
        kernel_size_4 = CSH.CategoricalHyperparameter(name='kernel_size_4', choices=[3,5,7])
        kernel_size_5 = CSH.CategoricalHyperparameter(name='kernel_size_5', choices=[3,5,7])

        filter_cond_1 = CS.GreaterThanCondition(num_filter_2, num_layer, 1)
        filter_cond_2 = CS.GreaterThanCondition(num_filter_3, num_layer, 2)
        filter_cond_3 = CS.GreaterThanCondition(num_filter_4, num_layer, 3)
        filter_cond_4 = CS.GreaterThanCondition(num_filter_5, num_layer, 4)

        kernel_cond_1 = CS.GreaterThanCondition(kernel_size_2, num_layer, 1)
        kernel_cond_2 = CS.GreaterThanCondition(kernel_size_3, num_layer, 2)
        kernel_cond_3 = CS.GreaterThanCondition(kernel_size_4, num_layer, 3)
        kernel_cond_4 = CS.GreaterThanCondition(kernel_size_5, num_layer, 4)

        cs.add_hyperparameters([num_layer,
                                num_filter_1,
                                num_filter_2,
                                num_filter_3,
                                num_filter_4,
                                num_filter_5,
                                kernel_size_1,
                                kernel_size_2,
                                kernel_size_3,
                                kernel_size_4,
                                kernel_size_5,
                                ])

        cs.add_conditions([filter_cond_1,
                           filter_cond_2,
                           filter_cond_3,
                           filter_cond_4,
                           kernel_cond_1,
                           kernel_cond_2,
                           kernel_cond_3,
                           kernel_cond_4,
                           ])

    else:
        raise ValueError('unknown model: ' + str(model))

    return cs


def transform_time(t, T, t0=None):
    """Logarithmic time scaling Transform for ALC """
    if t0 is None:
        t0 = T
    return np.log(1 + t / t0) / np.log(1 + T / t0)


def calculate_alc(timestamps, scores, start_time=0, time_budget=7200):
    """Calculate ALC """
    ######################################################
    # Transform X to relative time points
    timestamps = [t for t in timestamps if t <= time_budget + start_time]
    t0 = 60
    transform = lambda t: transform_time(t, time_budget, t0=t0)
    relative_timestamps = [t - start_time for t in timestamps]
    Times = [transform(t) for t in relative_timestamps]
    ######################################################
    Scores = scores.copy()
    # Add origin as the first point of the curve and end as last point
    ######################################################
    Times.insert(0, 0)
    Scores.insert(0, 0)
    Times.append(1)
    Scores.append(Scores[-1])
    ######################################################
    # Compute AUC using step function rule or trapezoidal rule
    alc = auc(Times, Scores)
    return alc


def get_configuration(dataset, model):
    '''Return BOHB's configuration (not model configuration!)'''
    cfg = {}
    cfg["code_dir"] = '/home/repo/autodl/starting_kit/code_submission'

    cfg["dataset"] = dataset
    cfg["model"] = model
    cfg["bohb_min_budget"] = 3  # budget as time (seconds)
    cfg["bohb_max_budget"] = 1200
    cfg["bohb_iterations"] = 10
    cfg["bohb_eta"] = 3
    cfg["bohb_log_dir"] = "./logs_new/" + dataset + '___' + model + '___' + str(int(time.time()))
    cfg["auc_splits"] = 10  # Unused

    #time_series_dataset_dir = "home/repo/autodl/starting_kit/sample_data"
    time_series_dataset_dir = os.path.join(os.getcwd(), 'sample_data')
    time_series_dataset = os.path.join(time_series_dataset_dir, dataset)

    if os.path.isdir(time_series_dataset):
        cfg['dataset_dir'] = time_series_dataset
    else:
        raise ValueError('Dataset does not exist: ' + str(time_series_dataset))

    return cfg


def calc_avg(score_list):
    score = sum(score_list) / len(score_list)

    print('--------- score list ---------')
    print(score_list)
    print('--------- score ---------')
    print(score)

    return score


def execute_run(cfg, config, budget):
    dataset_dir = cfg['dataset_dir']
    dataset = cfg['dataset']

    lower_path = os.path.join(dataset_dir, (dataset+'.data').lower())
    capital_path = os.path.join(dataset_dir, dataset+'.data')
    if os.path.exists(lower_path):
        D = AutoSpeechDataset(lower_path)
    else:
        D = AutoSpeechDataset(capital_path)

    # Get correct prediction shape
    num_examples_test = D.get_test_num()
    output_dim = D.get_class_num()
    correct_prediction_shape = (num_examples_test, output_dim)

    # Get train and test dataset
    D.read_dataset()
    D_train = D.get_train()  # this is a list: [train_data, train_labels]
    D_test = D.get_test()  # this is only test_data without labels

    # Initialize model
    M = Model(D_train, D_test, D.get_metadata(), cfg, config, budget)

    # lists to store timestamps and scores
    t_list = []
    score_list = []

    pred_queue = multiprocessing.Queue()
    time_queue = multiprocessing.Queue()

    # Train the model for given budget
    process = multiprocessing.Process(target=M.train_and_make_prediction, args=(pred_queue, time_queue))
    process.start()

    # wait for the budget to exhaust
    time.sleep(budget)

    # terminate process
    if process.is_alive():
        process.terminate()

    process.join()

    solution = get_solution(dataset_dir)

    # Get prediction from the queue and compute score.
    while not pred_queue.empty():
        prediction = pred_queue.get()
        t = time_queue.get()

        score = autodl_auc(solution, prediction)
        score_list.append(score)
        t_list.append(t)

    alc = calculate_alc(t_list, score_list, start_time=0, time_budget=7200)
    return alc


class BOHBWorker(Worker):
    def __init__(self, cfg, *args, **kwargs):
        super(BOHBWorker, self).__init__(*args, **kwargs)
        self.cfg = cfg
        print(cfg)

    def compute(self, config, budget, *args, **kwargs):
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('BUDGET: ' + str(budget))

        info = {}

        score = 0
        try:
            print('BOHB ON DATASET: ' + str(cfg["dataset"]))
            print('BOHB WITH MODEL: ' + str(cfg["model"]))
            score = execute_run(cfg=cfg, config=config, budget=budget)
        except Exception:
            status = traceback.format_exc()
            print(status)

        info[cfg["dataset"]] = score
        info['config'] = str(config)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print('----------------------------')
        print("END BOHB ITERATION")

        return {
            "loss": -score,
            "info": info
        }


def runBOHB(cfg):
    run_id = "0"

    # assign random port in the 30000-40000 range to avoid using a blocked port because of a previous improper bohb shutdown
    port = int(30000 + random.random() * 10000)

    ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=port)
    ns.start()

    w = BOHBWorker(cfg=cfg, nameserver="127.0.0.1", run_id=run_id, nameserver_port=port)
    w.run(background=True)

    result_logger = hpres.json_result_logger(
        directory=cfg["bohb_log_dir"], overwrite=True
    )

    bohb = BOHB(
        configspace=get_configspace(cfg['model']),
        run_id=run_id,
        min_budget=cfg["bohb_min_budget"],
        max_budget=cfg["bohb_max_budget"],
        eta=cfg["bohb_eta"],
        nameserver="127.0.0.1",
        nameserver_port=port,
        result_logger=result_logger,
    )

    res = bohb.run(n_iterations=cfg["bohb_iterations"])
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


if __name__ == "__main__":
    datasets = ['data02', 'data04']
    models = ['CNN_1D', 'ESN', 'LSTM', 'FCN']

    if len(sys.argv) == 3:      # parallel processing
        for arg in sys.argv[1:]:
            print(arg)

        id = int(sys.argv[1])
        tot = int(sys.argv[2])
        for i, dataset in enumerate(datasets):
            if (i-id)%tot != 0:
                continue
            for model in models:
                cfg = get_configuration(dataset, model)
                res = runBOHB(cfg)
    else:                       # serial processing
        for dataset in datasets:
            for model in models:
                cfg = get_configuration(dataset, model)
                res = runBOHB(cfg)