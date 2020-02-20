import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'ingestion'))
sys.path.append(os.path.join(os.getcwd(), 'scoring'))

import threading
import queue
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
import logging
import pickle
import torch


LOGGER = logging.getLogger(__name__)


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
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='steps_to_train', lower=10, upper=100))

    if model == 'LSTM':
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='last_n_outputs', lower=1, upper=50, log=False))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='num_layers', lower=1, upper=5, log=False))
        cs.add_hyperparameter((CSH.UniformIntegerHyperparameter(name='hidden_dim', lower=20, upper=100)))

    elif model == 'ESN':
        last_n_outputs = CSH.UniformIntegerHyperparameter(name='last_n_outputs', lower=10, upper=50, log=False)
        hidden_dim = CSH.UniformIntegerHyperparameter(name='hidden_dim', lower=200, upper=1000)
        sparsity = CSH.UniformFloatHyperparameter(name='sparsity', lower=0.75, upper=0.99, log=False)
        leak_rate = CSH.UniformFloatHyperparameter(name='leak_rate', lower=0.75, upper=1, log=False)

        cs.add_hyperparameters([last_n_outputs,
                                hidden_dim,
                                sparsity,
                                leak_rate,
                                ])

    elif model == 'FCN':
        num_layer = CSH.UniformIntegerHyperparameter(name='num_layers', lower=1, upper=5, log=False)

        num_filter_1 = CSH.UniformIntegerHyperparameter(name='num_filters_1', lower=16, upper=128, log=True)
        num_filter_2 = CSH.UniformIntegerHyperparameter(name='num_filters_2', lower=16, upper=128, log=True)
        num_filter_3 = CSH.UniformIntegerHyperparameter(name='num_filters_3', lower=16, upper=128, log=True)
        num_filter_4 = CSH.UniformIntegerHyperparameter(name='num_filters_4', lower=16, upper=128, log=True)
        num_filter_5 = CSH.UniformIntegerHyperparameter(name='num_filters_5', lower=16, upper=128, log=True)

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

        num_filter_1 = CSH.UniformIntegerHyperparameter(name='num_filters_1', lower=16, upper=128, log=True)
        num_filter_2 = CSH.UniformIntegerHyperparameter(name='num_filters_2', lower=16, upper=128, log=True)
        num_filter_3 = CSH.UniformIntegerHyperparameter(name='num_filters_3', lower=16, upper=128, log=True)
        num_filter_4 = CSH.UniformIntegerHyperparameter(name='num_filters_4', lower=16, upper=128, log=True)
        num_filter_5 = CSH.UniformIntegerHyperparameter(name='num_filters_5', lower=16, upper=128, log=True)

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
    cfg["bohb_min_budget"] = 20  # budget as time (seconds)
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

    # Get data
    lower_path = os.path.join(dataset_dir, (dataset+'.data').lower())
    capital_path = os.path.join(dataset_dir, dataset+'.data')
    if os.path.exists(lower_path):
        D = AutoSpeechDataset(lower_path)
    else:
        D = AutoSpeechDataset(capital_path)

    # Get train and test dataset
    D.read_dataset()
    D_train = D.get_train()  # this is a list: [train_data, train_labels]
    D_test = D.get_test()  # this is only test_data without labels

    # Initialize model with given configuration.
    M = Model(D_train, D_test, D.get_metadata(), cfg, config, budget)

    # lists to store timestamps and scores
    t_list = []
    score_list = []

    pred_queue = queue.Queue()
    time_queue = queue.Queue()

    # Sends done signal
    running = threading.Event()
    running.set()

    # Train the model for given budget.
    thread = threading.Thread(target=M.train_and_make_prediction, args=(pred_queue, time_queue, running))
    thread.start()

    # wait for the budget to exhaust
    time.sleep(budget)

    # Stop running the thread
    running.clear()
    thread.join()

    solution = get_solution(dataset_dir)

    # Get prediction from the queue and compute score.
    if pred_queue.qsize() != time_queue.qsize():
        raise ValueError("Got unequal amount of predictions and timestamps!")

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
            print("exec run done")
        except Exception:
            status = traceback.format_exc()
            print("STATUS: ", status)

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

    res = bohb.run(   n_iterations=cfg["bohb_iterations"])
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


if __name__ == "__main__":
    datasets = ['data01', 'data02', 'data03', 'data04', 'data05']
    models = ['ESN', 'CNN_1D', 'LSTM', 'FCN']

    if len(sys.argv) == 3:      # parallel processing
        for arg in sys.argv[1:]:
            print(arg)

        print("Cuda available: ", torch.cuda.is_available())

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
