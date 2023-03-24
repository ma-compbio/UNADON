from utils import *
from attn import *

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import time

from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Function


class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return torch.transpose(x, -1, -2)


# reference: https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
class GradientReverse(torch.autograd.Function):
    scale = 5
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()
    
def grad_reverse(x, scale=5):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, 50))
        self.layers.append(nn.Linear(50, 50))
        self.layers.append(nn.Linear(50, 4))


    def forward(self,x):
        out = grad_reverse(x)
        for layer in self.layers:
            out = layer(out)
        out = nn.Softmax(dim = -1)(out)
        out = out.permute(0, 2, 1)

        return out


class PrepNet(nn.Module):
    """
    The data processing subnetwork for each data modality

    Args:
        input_dim: the expected dimension of the input feature vector
        hidden_dim: the number of hidden units in each layer
        num_layers: the number of hidden layers
        dropout: the dropout rate
   
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        
        super().__init__()
        dropout = 0.5
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(Transpose())
        self.layers.append(nn.BatchNorm1d(hidden_dim, track_running_stats = False))
        self.layers.append(Transpose())
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))

        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(Transpose())
            self.layers.append(nn.BatchNorm1d(hidden_dim, track_running_stats = False))
            self.layers.append(Transpose())
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        # CNN can be used as a preprocessing module but not used in this work
        # self.layers.append(nn.Conv1d(in_channels = dim, out_channels = dim, kernel_size = 3, padding = 1))
        # # self.layers.append(nn.BatchNorm1d(dim, track_running_stats = False))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.Dropout(dropout))


        self.output = nn.Linear(hidden_dim,hidden_dim)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = self.output(out)
        return out


class UNADON(nn.Module):
    """
    The main UNADON neural network based on transformer

    Args:
        input_dim: the expected dimension of the input feature vector
        seq_end: the end position of the sequence features
        dense_dim: the number of hidden units in the dense layer for the data processing subnetwork
        dense_num_layers: the number of dense layers for the data processing subnetwork
        nhead: the number of heads in the multi-head attention layer
        attn_hidden_dim: the hidden dimension for the transformer encoder module
        attn_num_layers: the number of transformer encoder layers
        dropout: the dropout rate
   
    """
    def __init__(self, input_dim, seq_end, dense_dim, dense_num_layers, nhead,
        attn_hidden_dim, attn_num_layers, dropout):
        super().__init__()

        # Define where to split the two features
        self.seq_end = seq_end

        # If both seq and epi are included
        if input_dim > seq_end:
            self.seqNet = PrepNet(seq_end, dense_dim, dense_num_layers, dropout)
            self.histNet = PrepNet(input_dim - seq_end, dense_dim, dense_num_layers, dropout)
        # For sequence-only model
        elif input_dim == seq_end:
            self.seqNet = PrepNet(seq_end, 2 * dense_dim, dense_num_layers, dropout)
            self.histNet = None 
        # For histone-only model
        else:
            self.seqNet = None
            self.seq_end = 0
            self.histNet = PrepNet(input_dim - self.seq_end, 2 * dense_dim, dense_num_layers, dropout)


        # Domain classifier for cross-cell-type training only
        self.domain_clf = DomainClassifier(16)
        self.FC_merge = nn.Linear(2 * dense_dim, 2 * dense_dim)
        self.FC1 = nn.Linear(2 * dense_dim, 16)

        self.encoder_layer =  TransformerEncoderLayer(d_model = 2 * dense_dim, nhead=nhead,\
            dim_feedforward = attn_hidden_dim, batch_first = True, dropout = dropout,\
            activation = 'relu')
        self.transformer = TransformerEncoder(self.encoder_layer, attn_num_layers)
        
        self.output = nn.Linear(16, 1)

        # self.pe_base = 1 / 10000 ** (torch.arange(0, 64, 2) / 64)

    def forward(self, x):
        seq = x[:,:,:self.seq_end]
        hist = x[:,:,self.seq_end:]
        if self.histNet == None: # Sequence-only
            out = self.seqNet(seq)
        elif self.seqNet == None: # Histone-only
            out = self.histNet(hist)
        else:
            seq_out, hist_out = self.seqNet(seq), self.histNet(hist)
            out = torch.cat((seq_out, hist_out), dim = -1)
            out = self.FC_merge(out)
        # out = out  + 0.1 * self.get_positional_encoding(out.shape, out.get_device()) # only for absolute positional encoding

        # (batch, seq, feature) to (seq, batch, feature)
        out = torch.transpose(out, 0, 1)
        (out, attn) = self.transformer(out)

        # (seq, batch, feature) to (batch, seq, feature) 
        out = torch.transpose(out, 0, 1)
        out = self.FC1(out)
        out = nn.ReLU()(out)
        domain = self.domain_clf(out)
        out = self.output(out)

        out = nn.Flatten(-2,-1)(out)
        out = nn.Tanh()(out)
        return (out, domain)

    # For absolute positional encoding only. Not used in this work.
    # def get_positional_encoding(self, shape, device):
    #     pos = torch.arange(shape[1])
    #     pe1 = torch.sin(torch.einsum('i,j->ij', pos, self.pe_base))
    #     pe2 = torch.cos(torch.einsum('i,j->ij', pos, self.pe_base))
    #     pe = torch.cat((pe1[:,:,None],pe2[:,:,None]), axis = -1)
    #     pe = pe.reshape((pe.shape[0],pe.shape[1] * 2)) 
    #     pe_all = pe.unsqueeze(0).repeat((shape[0],1,1)) 
    #     # pe_all = torch.rand((shape[0],shape[1],shape[2]))
    #     return pe_all.to(device)