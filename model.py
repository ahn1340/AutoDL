import threading
import queue
import librosa
from sklearn.utils import shuffle
import json
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import models
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D
from tensorflow.python.keras.layers import MaxPooling2D,BatchNormalization
from tensorflow.python.keras.preprocessing import sequence

from keras.backend.tensorflow_backend import set_session


# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable
import time
from networks import CNN_1D, FCN, ESN, LSTM
#from ingestion.dataset import AutoSpeechDataset

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
"""
This script is used for training the configurable models for BOHB.
"""


################################ Helper Functions #################################
def extract_mfcc(data,sr=16000):
    results = []
    for d in data:
        r = librosa.feature.mfcc(d,sr=16000,n_mfcc=13)
        r = r.transpose()
        results.append(r)
    return results

def pad_seq(data,pad_len):
    return sequence.pad_sequences(data,maxlen=pad_len,dtype='float32',padding='pre')

# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)

def accuracy(solution, prediction):
    """Get accuracy of 'prediction' w.r.t true labels 'solution'."""
    epsilon = 1e-15
    # normalize prediction
    prediction_normalized = \
        prediction / (np.sum(np.abs(prediction), axis=1, keepdims=True) + epsilon)
    return np.sum(solution * prediction_normalized) / solution.shape[0]

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_data_loader(train_X, train_y, batch_size=64):
    """Return an iterable train_loader. It is cyclic, in case the itration reaches its limit"""
    train_X = torch.Tensor(train_X)
    train_y = torch.Tensor(train_y)
    train = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    return iter(cycle(train_loader))
################################ Dataset Object #################################


class Model(object):

    def __init__(self, D_train, D_test, metadata, cfg, config, budget, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 7,
             "train_num": 428,
             "test_num": 107,
             "time_budget": 1800}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

        self.birthday = time.time()
        self.total_train_time = 0
        self.cumulated_num_steps = 0
        self.cumulated_num_tests = 0
        self.estimated_time_per_step = None
        self.total_train_time = 0
        self.total_test_time = 0
        self.estimated_time_test = None
        self.trained = False
        self.done_training = False
        self.cfg = cfg
        self.config = config
        self.budget = budget

        # Create dataloader
        train_data, train_labels = D_train
        train_features = extract_mfcc(train_data)
        test_features = extract_mfcc(D_test)

        train_max_len = max([len(_) for _ in train_features])
        test_max_len = max([len(_) for _ in test_features])
        max_len = max(train_max_len, test_max_len)  # for CNN variants we need max sequence length in advance

        train_features = pad_seq(train_features, max_len)  # padding at the beginning of the input
        test_features = pad_seq(test_features, max_len)

        self.train_loader = get_data_loader(train_features, train_labels, batch_size=config['batch_size'])
        self.test_features = test_features  # we don't make a dataloader for test samples

        num_mfcc_feature = 13  # fixed for our experiment
        self.input_dim = (max_len, num_mfcc_feature)  # Input dim: (time, feature_length)
        self.output_dim = metadata['class_num']

        if cfg['model'] == 'LSTM':
            self.pytorchmodel = LSTM(self.input_dim, self.output_dim, config)
        elif cfg['model'] == 'ESN':
            self.pytorchmodel = ESN(self.input_dim, self.output_dim, config)
        elif cfg['model'] == 'CNN_1D':
            self.pytorchmodel = CNN_1D(self.input_dim, self.output_dim, config)
        elif cfg['model'] == 'FCN':
            self.pytorchmodel = FCN(self.input_dim, self.output_dim, config)
        else:
            raise ValueError('Unknown model: ', + str(cfg['model']))
        ##########################################################


    def train_and_make_prediction(self, pred_queue, time_queue, running):
        """
        Idea: for given budget, do train for 20 steps and call test() to make prediction.
        Repeat this until the budget is exhausted.
        queue: Queue object to store Y_pred to return to the BOHB run
        """
        start_time = time.time()

        while running.is_set():

            self.train()
            Y_pred = self.test()

            correct_prediction_shape = (self.metadata['test_num'], self.output_dim)
            prediction_shape = tuple(Y_pred.shape)

            if prediction_shape != correct_prediction_shape:
                raise ValueError(
                    "Bad prediction shape! Expected {} but got {}." \
                        .format(correct_prediction_shape, prediction_shape)
                )

            pred_queue.put(Y_pred)
            time_queue.put(time.time() - start_time)

        # Delete the last prediction and the timestamp, because that is made after the
        # last while loop, during the process of which the budget is exhausted.
        #TODO: need to delete the last one in, not the first one.
        #TODO: Or, we can use while True, and then only add to queue when running.is_set().
        pred_queue.get()
        time_queue.get()



    def train(self):
        """
        model training on train_dataset.
        """
        print("Begin training...")
        steps_to_train = self.config['steps_to_train']

        criterion = nn.BCEWithLogitsLoss()  # For multilabel classification this should be used
        if self.config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.pytorchmodel.parameters(),
                                        lr=self.config['lr'],
                                        weight_decay=self.config['weight_decay'],
                                        )
        elif self.config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.pytorchmodel.parameters(),
                                         lr=self.config['lr'],
                                         weight_decay=self.config['weight_decay'],
                                        )

        self.trainloop(criterion, optimizer, steps_to_train)

    def trainloop(self, criterion, optimizer, steps):
        '''
        # PYTORCH
        Trainloop function does the actual training of the model
        3) trains the model with the Tensors for given no of steps.
        '''
        accs = []
        losses = []

        for i in range(steps):

            features, labels = next(self.train_loader)
            features = torch.Tensor(features)
            labels = torch.Tensor(labels)

            if torch.cuda.is_available():
                # No Cuda for now
                # images = images.float().cuda()
                # labels = labels.long().cuda()
                images = features.float()
                labels = labels.float()
            else:
                images = features.float()
                labels = labels.float()
            optimizer.zero_grad()

            with open("input_train.txt", "a") as f:
                f.write(str(labels))

            # model output (log probability)
            log_ps = self.pytorchmodel(images)

            # Create train prediction for computing accuracy
            preds = []
            loss = criterion(log_ps, labels)
            top_p, top_class = log_ps.topk(1, dim=1)
            preds.append(top_class.cpu().numpy())
            preds = np.concatenate(preds)
            onehot_preds = np.squeeze(np.eye(self.output_dim)[preds.reshape(-1)])
            train_accuracy = accuracy(labels.numpy(), onehot_preds)

            # print train loss and accuracy
            #if i % 5 == 0:
            #    print("remaining step: {}".format(steps - i))
            #    print("loss: ", float(loss))
            #    print("train accuracy: ", accuracy(labels.numpy(), onehot_preds))
            loss.backward()
            optimizer.step()

            accs.append(train_accuracy)
            losses.append(loss.detach().numpy())

        print("Train accuracy: {}, loss: {}".format(np.mean(train_accuracy), np.mean(losses)))


    #PYTORCH
    def test(self, remaining_time_budget=None):
        predictions = self.testloop(self.test_features)
        return predictions


    def testloop(self, test_x):
        '''
        # PYTORCH
        testloop uses testdata to test the pytorch model and return onehot prediciton values.
        '''
        preds = []
        with torch.no_grad():
            self.pytorchmodel.eval()
            test_x = torch.Tensor(test_x)

            if torch.cuda.is_available():
                # no Cuda for now
                #images = images.float().cuda()
                test_x = test_x.float()
            else:
                test_x = test_x.float()


            log_ps = self.pytorchmodel(test_x)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            preds.append(top_class.cpu().numpy())
            preds = np.concatenate(preds)
            onehot_preds = np.squeeze(np.eye(self.output_dim)[preds.reshape(-1)])
            return onehot_preds
