import numpy as np
from utils import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import h5py
from utils import *
import time

import xgboost as xgb

'''

Model architectures for the baseline models

'''


# xgboost
def xgboost_model(n_estimators, early_stopping_rounds, random_state):
    model = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators = n_estimators, \
        seed = random_state, tree_method='gpu_hist',early_stopping_rounds=early_stopping_rounds)

# Dense neural network
class DNN(nn.Module):
    '''
    Neural network with only dense layers
    Args:
        input_dim: the expected dimension of the input feature vector
        hidden_dim: the list of hidden units within each hidden layer
        The number of hidden layers is implicitly defined by hidden_dim
        dropout: the dropout rate
    '''
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim[0]))
        for i in range(1, len(hidden_dim)):
            self.layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        self.output = nn.Linear(hidden_dim[-1],1)


    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = self.output(out)
        out = nn.Tanh()(out)
        out = nn.Flatten(-2,-1)(out)

        return out

# Convolutional neural network
class CNN(nn.Module):
    '''
    Convolutional Neural Network
    Args:
        input_dim: the expected dimension of the input feature vector
        hidden_dim: the number of hidden units (channels)
        num_layers: the number of convolutional layers
        kernel_size: the size of the kernel
        dropout: the dropout rate
    '''
    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(Transpose())
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))

        for i in range(num_layers - 1):
            self.layers.append(nn.Conv1d(in_channels = hidden_dim, out_channels = hidden_dim, \
                kernel_size = kernel_size, padding = kernel_size // 2))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        

        self.layers.append(Transpose())
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(Transpose())
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))

        self.output = nn.Linear(hidden_dim, 1)

        self.weight_initialization()

    def weight_initialization(self):
        for layer in self.layers:
            try:
                torch.nn.init.kaiming_uniform_(layer.weight)
                layer.bias.data.fill_(0.0)
            except:
                pass

        torch.nn.init.xavier_normal_(self.output.weight)

    def forward(self, x):
        out = x
        # print(out.shape)
        for layer in self.layers:
            out = layer(out)
        out = out.permute(0,2,1)
        out = self.output(out)

        out = nn.Flatten(-2,-1)(out)
        out = nn.Tanh()(out)

        return out



def output_dimension(i, p, k, d):
    return i + 2 * p - k - (k-1) * (d-1) + 1 


    

class DilatedCNN(nn.Module):
    '''
    Convolutional Neural Network with dilation
    Args:
        input_dim: the expected dimension of the input feature vector
        hidden_dim: the number of hidden units (channels)
        num_conv_layers: the number of convolutional layers
        kernel_size: the size of the kernel
        dilation: list of dilation rates
        dropout: the dropout rate

    
    '''

    def __init__(self, input_dim, hidden_dim, num_conv_layers, kernel_size, dilation, dropout):
        super().__init__()
        # We found that adding a dense layer at the beginning greatly booosted the performance
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(Transpose())
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))


        for i in range(num_conv_layers):
            self.layers.append(nn.Conv1d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = kernel_size, padding = kernel_size // 2))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        
        for d in dilation:
            self.layers.append(nn.Conv1d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = kernel_size, padding = self.padding_num(kernel_size, d), dilation = d))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
    

        self.layers.append(Transpose())
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(Transpose())
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))

        self.output = nn.Linear(hidden_dim, 1)

        self.weight_initialization()

    # Compute the number of padding required for preserving the dimension.  
    def padding_num(self, k, d):
        return (k + (k - 1) * (d - 1) - 1) // 2

    def weight_initialization(self):
        for layer in self.layers:
            try:
                torch.nn.init.kaiming_uniform_(layer.weight)
                layer.bias.data.fill_(0.0)
            except:
                pass

        torch.nn.init.xavier_normal_(self.output.weight)

    def forward(self, x):
        out = x
        # print(out.shape)
        for layer in self.layers:
            out = layer(out)
        out = out.permute(0,2,1)
        out = self.output(out)

        out = nn.Flatten(-2,-1)(out)
        out = nn.Tanh()(out)

        return out
