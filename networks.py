import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

import numpy as np
"""
Configurable versions of the models
"""

############################### Models #######################################3
class CNN_1D(nn.Module):
    """
    Applies convolution over the entire feature dimension.
    for CNNS the length of the entire sequence of both train and test dataset
    must be determined before initializiation of the network.
    """
    def __init__(self, input_dim, output_dim, config):
        """
        :param input_dim: 2d list_like object with shape (time, feature)
        :param output_dim: number classes
        """
        super(CNN_1D, self).__init__()
        print(config)

        time_dim, feat_dim = input_dim  # shape = (T, F)

        num_layers = config['num_layers']
        num_filters_per_layer = [config[k] for k in sorted(config.keys()) if k.startswith('num_filters')]
        num_input_filters = input_dim[1]
        layers = OrderedDict()

        for i, num_filters in enumerate(num_filters_per_layer):
            # Convolution always preserves original size.
            layers['conv'+str(i+1)] = nn.Conv1d(num_input_filters,
                                                num_filters,
                                                kernel_size=config['kernel_size_'+str(i+1)],
                                                padding=(config['kernel_size_'+str(i+1)] - 1) // 2,
                                                )
            layers['batchnorm'+str(i+1)] = nn.BatchNorm1d(num_filters)
            layers['relu'+str(i+1)] = nn.ReLU()
            # Maxpool always halves the size of time dimension
            layers['pool'+str(i+1)] = nn.MaxPool1d(2)
            num_input_filters = num_filters

        layers['flatten'] = nn.Flatten(start_dim=1)
        layers['linear'] = nn.Linear(num_input_filters * (time_dim // 2**num_layers), output_dim)

        self.model = nn.Sequential(layers)


    def forward(self, x):
        # x is of shape (N, T, F). Since F is considered as the channel dimension and
        # we convolve over the time dimension, swap the axis of T and F.
        x = x.permute(0, 2, 1)
        x = self.model(x)

        return x


class FCN(nn.Module):
    """
    Fully Convolutional network.
    The paper "Time Series Classification from Scratch with Deep Neutal Networks: a Strong Baseline"
    states that FCN beat ResNet and MLP in various TSC tasks.
    """
    def __init__(self, input_dim, output_dim, config):
        super(FCN, self).__init__()
        self.time_dim = input_dim[0]
        self.feat_dim = input_dim[1]
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(self.feat_dim, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, self.output_dim, kernel_size=3, padding=1)

        self.batchnorm1 = nn.BatchNorm1d(32)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(self.output_dim)

        self.GlobalAvgPooling = nn.AvgPool1d(self.time_dim)
        self.linear = nn.Linear(self.output_dim, self.output_dim)


    def forward(self, x):
        # X is of shape (N, T, F). Since F is considered as the channel dimension and
        # we convolve over the time dimension, swap the axis of T and F.
        batch_size = x.size()[0]
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)

        # Apply GAP such that the size of final convolution = output_dim
        x = self.GlobalAvgPooling(x)
        x = x.view(x.size()[0], -1)
        out = self.linear(x)

        return out



class ESN(nn.Module):
    """
    Echo State Network for Time Series Classification
    """
    def __init__(self, input_dim, output_dim, config):
        super(ESN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
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
    def __init__(self, input_dim, output_dim, config):
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
