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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


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
################################ Helper Functions #################################


############################### Models #######################################3
class MLP_baseline(nn.Module):
    """
    Simple feed_forward network for baseline.
    """
    def __init__(self, input_dim, output_dim):
        """
        :param input_size: list-like of shape (T, F)
        :param output_size: number of classes
        """
        super(MLP_baseline, self).__init__()
        self.input_size = input_dim[0] * input_dim[1]
        self.linear1 = nn.Linear(self.input_size, 1000)
        self.linear2 = nn.Linear(1000, 600)
        self.linear3 = nn.Linear(600, 200)
        self.linear3 = nn.Linear(200, output_dim)


    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        out = F.relu(x)

        return out


class CNN_1D(nn.Module):
    """
    Applies convolution over the entire feature dimension.
    for CNNS the length of the entire sequence of both train and test dataset
    must be determined before initializiation of the network.
    """
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: 2d list_like object with shape (time, feature)
        :param output_dim: number classes
        """
        super(CNN_1D, self).__init__()

        self.time_dim = input_dim[0]
        self.feat_dim = input_dim[1]  # here, feature dimension is considered as channel dimension.
        self.output_dim = output_dim

        # assumes that input is of shape (N, T, F)
        self.conv1 = nn.Conv1d(self.feat_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.linear1 = nn.Linear(128 * (self.time_dim // 4), 128)
        self.linear2 = nn.Linear(128, self.output_dim)

    def forward(self, x):
        # X is of shape (N, T, F). Since F is considered as the channel dimension and
        # we convolve over the time dimension, swap the axis of T and F.
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = x.view(x.size()[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)

        return x


class FCN(nn.Module):
    """
    Fully Convolutional network.
    The paper "Time Series Classification from Scratch with Deep Neutal Networks: a Strong Baseline"
    states that FCN beat ResNet and MLP in various TSC tasks.
    """
    def __init__(self, input_dim, output_dim):
        self.time_dim = input_dim[0]
        self.feat_dim = input_dim[1]

        self.conv1 = nn.Conv1d(self.feat_dim, 128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        #TODO

    def forward(self, x):
        #TODO
        pass


class ESN(nn.Module):
    """
    Echo State Network for Time Series Classification
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ESN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sparsity = 0.8
        self.last_n_outputs = 20  # Hyperparameter

        self.w_in = torch.randn(input_dim, hidden_dim) * 0.2

        # W_r needs sparsity, spectral radius < 1, zero mean standard gaussian distributed non-zero elements.
        w_r = np.random.randn(hidden_dim, hidden_dim)
        w_r = w_r / np.max(w_r)
        # zero out elements
        w_r = np.where(np.random.random(w_r.shape) < self.sparsity, 0, w_r)
        # Spectral norm
        max_abs_eigval = np.max(np.abs(np.linalg.eigvals(w_r)))
        w_r = w_r / max_abs_eigval

        self.w_r = torch.Tensor(w_r)

        # readout layer
        self.linear1 = nn.Linear(hidden_dim * self.last_n_outputs, output_dim)

    def forward(self, x):
        # goes through the resorvoir
        #print(x.shape)
        batch_size = x.size()[0]
        time_steps = x.size()[1]  # assumes input is shape (N, T, F)
        alpha = 0.9 # Leaky unit thing hyperparameter. To be Hyperparametrized

        # Add 1 to feature row as done in many ESN papers. Why? don't know...
        #ones = torch.Tensor(batch_size, time_steps, 1)
        #x = torch.cat((x, ones), 2)

        # initialize empty h(0)
        h_t = torch.zeros(batch_size, self.hidden_dim)
        # list to store last n outputs
        out_list = []

        for t in range(time_steps):
            h_t_hat = F.tanh(torch.add(torch.matmul(x[:, t, :], self.w_in), torch.matmul(h_t, self.w_r)))
            if t == 0:
                h_t = h_t_hat
            else:
                h_t = (1 - alpha) * h_t + alpha * h_t_hat
            if time_steps - t <= self.last_n_outputs:
                out_list.append(h_t)

        # concatenate output list
        out_list = torch.cat(out_list, dim=1)
        out = self.linear1(out_list)
        return out


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=7, num_layers=2):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.last_n_outputs = 20  # Hyperparameter

        # Define LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define output layer
        self.linear = nn.Linear(self.hidden_dim * self.last_n_outputs, output_dim)

        # hidden state
        self.hidden = None

    def init_hidden(self, batch_size):
        # This is what we will initialize hidden states as.
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        h0 = Variable(h0)
        c0 = Variable(c0)

        return (h0, c0)

    def forward(self, x):

        batch_size = x.shape[0]
        # Create initial hidden and cell state.
        self.hidden = self.init_hidden(batch_size)
        x, _ = self.lstm(x, self.hidden)
        # take the last n output of LSTM
        output = torch.flatten(x[:, -(self.last_n_outputs+1):-1, :], start_dim=1)

        out = self.linear(output)
        return out
############################### Models #######################################3




class Model(object):

    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
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

        ################## LSTM ###################################
        input_dim = 13
        output_dim = 7
        hidden_dim = 20
        num_layers = 1
        self.output_dim = output_dim

        self.pytorchmodel = LSTM(input_dim, hidden_dim, output_dim, num_layers)
        ##########################################################
        ################# CNN_1D #######################################
        #input_dim = (281, 13)
        ## Sequence length (mfcc features)
        ## DEMO and data02 = 281
        ## data04 = 157
        ## data01 = ?
        ## data05 = ?
        ## data03 = ?
        #output_dim = 7
        #self.output_dim = output_dim

        #self.pytorchmodel = CNN_1D(input_dim, output_dim)
        ##########################################################
        ################## ESN ###################################
        #input_dim = 13
        #output_dim = 7
        #hidden_dim = 400
        #num_layers = 4
        #self.output_dim = output_dim

        #self.pytorchmodel = ESN(input_dim, hidden_dim, output_dim, num_layers)
        ##########################################################
        ##########################################################

    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.
        
        :param train_dataset: tuple, (x_train, y_train)
            train_x: list of vectors, input train speech raw data.
            train_y: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        steps_to_train = self.get_steps_to_train(remaining_time_budget)

        if steps_to_train <= 0:
            print("Not enough time remaining for training. " +
                  "Estimated time for training per step: {:.2f}, " \
                  .format(self.estimated_time_per_step) +
                  "but remaining time budget is: {:.2f}. " \
                  .format(remaining_time_budget) +
                  "Skipping...")
            self.done_training = True

        if self.done_training:
            return
        train_x, train_y = train_dataset
        train_x, train_y = shuffle(train_x, train_y)

        fea_x = extract_mfcc(train_x)
        max_len = max([len(_) for _ in fea_x])
        # Hack for CNN
        #max_len = 281  # For DEMO and data02
        #max_len = 157  # For data04. train = test
        fea_x = pad_seq(fea_x, max_len)
        num_class = self.metadata['class_num']
        dataset = [fea_x, train_y]


        # PYTORCH
        # Training loop inside
        train_start = time.time()
        #criterion = nn.NLLLoss()
        criterion = nn.BCEWithLogitsLoss()  # For multilabel classification this should be used
        optimizer = torch.optim.Adam(self.pytorchmodel.parameters(), lr=1e-3, weight_decay=0.01)
        SGD_optimizer = torch.optim.SGD(self.pytorchmodel.parameters(), lr=1e-2)
        self.trainloop(criterion, optimizer, dataset, steps=steps_to_train)
        train_end = time.time()

        # Update for time budget managing
        train_duration = train_end - train_start
        self.total_train_time += train_duration
        self.cumulated_num_steps += steps_to_train
        self.estimated_time_per_step = self.total_train_time / self.cumulated_num_steps


    def get_steps_to_train(self, remaining_time_budget):
        """Get number of steps for training according to `remaining_time_budget`.

        The strategy is:
          1. If no training is done before, train for 10 steps (ten batches);
          2. Otherwise, estimate training time per step and time needed for test,
             then compare to remaining time budget to compute a potential maximum
             number of steps (max_steps) that can be trained within time budget;
          3. Choose a number (steps_to_train) between 0 and max_steps and train for
             this many steps. Double it each time.
        """
        if not remaining_time_budget:  # This is never true in the competition anyway
            remaining_time_budget = 1200  # if no time limit is given, set to 20min

        if not self.estimated_time_per_step:
            steps_to_train = 10
        else:
            if self.estimated_time_test:
                tentative_estimated_time_test = self.estimated_time_test
            else:
                tentative_estimated_time_test = 50  # conservative estimation for test
            max_steps = int((remaining_time_budget - tentative_estimated_time_test) / self.estimated_time_per_step)
            max_steps = max(max_steps, 1)
            if self.cumulated_num_tests < np.log(max_steps) / np.log(2):
                steps_to_train = int(2 ** self.cumulated_num_tests)  # Double steps_to_train after each test
            else:
                steps_to_train = 0
        return steps_to_train
        #return 100


    def trainloop(self, criterion, optimizer, dataset, steps):
        '''
        # PYTORCH
        Trainloop function does the actual training of the model
        1) it gets the X, y from tensorflow dataset.
        2) convert X, y to CUDA
        3) trains the model with the Tensors for given no of steps.
        '''

        print("Training steps: {}".format(steps))
        #images, labels = dataset
        #batch_size = 64

        for i in range(steps):
            #images = torch.Tensor(images[i*batch_size: (i+1)*batch_size])
            #labels = torch.Tensor(labels[i*batch_size: (i+1)*batch_size])
            images, labels = dataset
            images = torch.Tensor(images)
            labels = torch.Tensor(labels)

            if torch.cuda.is_available():
                # No Cuda for now
                # images = images.float().cuda()
                # labels = labels.long().cuda()
                images = images.float()
                labels = labels.float()
            else:
                images = images.float()
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

            # print train loss and accuracy
            if i % 5 == 0:
                print("#"*10)
                print("remaining step: {}".format(steps - i))
                print("loss: ", loss)
                print("train accuracy: ", accuracy(labels.numpy(), onehot_preds))
                print("#"*10)
            loss.backward()
            optimizer.step()


    #PYTORCH
    def test(self, test_x, remaining_time_budget=None):
        fea_x = extract_mfcc(test_x)
        max_len = max([len(_) for _ in fea_x])
        fea_x = pad_seq(fea_x, max_len)
        #print("max_len for test_x: ",max_len)

        if self.done_training:
            print("done training")
            return None

        test_begin = time.time()
        if remaining_time_budget and self.estimated_time_test and\
            self.estimated_time_test > remaining_time_budget:
          print("Not enough time for test. " +\
                "Estimated time for test: {:.2e}, ".format(self.estimated_time_test) +\
                "But remaining time budget is: {:.2f}. ".format(remaining_time_budget) +\
                "Stop train/predict process by returning None.")
          return None

        msg_est = ""
        if self.estimated_time_test:
          msg_est = "estimated time: {:.2e} sec.".format(self.estimated_time_test)
        print("Begin testing...", msg_est)

        # PYTORCH
        predictions = self.testloop(fea_x)

        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin
        self.total_test_time += test_duration
        self.cumulated_num_tests += 1
        self.estimated_time_test = self.total_test_time / self.cumulated_num_tests
        print("[+] Successfully made one prediction. {:.2f} sec used. ".format(test_duration) +\
              "Total time used for testing: {:.2f} sec. ".format(self.total_test_time) +\
              "Current estimated time for test: {:.2e} sec.".format(self.estimated_time_test))
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