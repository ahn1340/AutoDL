"""
Script that goes through the BOHB results and extracts the best configuration
"""
import os
import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

datasets = ['espeak-starcraft-words',
            #'starcraft',
            'music_genre',
            'sick_no_sick',
            #'flickr',
            'music-speech',
            'music_env_speech_cleaned',
            'google_audio_set',
            'number_mnist',
            'timit_accent',
            'urbansound',
            'env-sound',
            'acoustic',
            'espeak-gender',
            'espeak-numbers',
            'librispeech',
            #'spokenlanguage',
            # 'data01',  # Test datasets
            # 'data02',
            # 'data03',
            # 'data04',
            # 'data05',
            ]
datasets = [#'espeak-starcraft-words',
            ##'starcraft',
            #'music_genre',
            #'sick_no_sick',
            ##'flickr',
            #'music-speech',
            #'music_env_speech_cleaned',
            #'google_audio_set',
            #'number_mnist',
            #'timit_accent',
            #'urbansound',
            #'env-sound',
            #'acoustic',
            #'espeak-gender',
            #'espeak-numbers',
            #'librispeech',
            #'spokenlanguage',
             'data01',  # Test datasets
             'data02',
             'data03',
             'data04',
             'data05',
            ]


def extract_best_config(dataset):
    """
    Gets the bohb results of the given dataset and return the best architecture and configuration.
    :param dataset: string
    :return:
    incumbent_model: string, one of ['ESN', 'CNN_1D', 'LSTM', 'FCN']
    incumbent_config: configuration object
    """
    models = ['ESN', 'CNN_1D', 'LSTM', 'FCN']
    result_dir = 'logs_sample_dataset'

    # BOHB results for each model&dataset combination
    results = [os.path.join(result_dir, name) for name in os.listdir(result_dir)]

    results_current_dataset = [r for r in results if dataset in r]

    # to store the losses of each and compare the results later
    incumbent_configs = []
    incumbent_losses = []

    for model in models:
        result_folder = [r for r in results_current_dataset if model in r]
        result = result_folder[0]

        # load the example run from the log files
        result = hpres.logged_results_to_HBS_result(result)

        # get the 'dict' that translates config ids to the actual configurations
        id2conf = result.get_id2config_mapping()

        # get incumbent id
        inc_id = result.get_incumbent_id()

        # get result of the incumbent
        inc_runs = result.get_runs_by_id(inc_id)
        inc_run = inc_runs[-1]

        inc_config = id2conf[inc_id]['config']  # best config
        inc_loss = inc_run.loss  # loss of the best config

        incumbent_configs.append(inc_config)
        incumbent_losses.append(inc_loss)

    min_loss, idx = min((val, idx) for (idx, val) in enumerate(incumbent_losses))
    incumbent_model = models[idx]
    incumbent_config = incumbent_configs[idx]
    print("################################### Dataset {} #######################################".format(dataset))
    print("Best performing model: {}".format(incumbent_model))
    print("Configuration: {}".format(incumbent_config))
    print("Score: {}".format(-min_loss))
    print("\n")

    return incumbent_model, incumbent_config

for dataset in datasets:
    extract_best_config(dataset)

