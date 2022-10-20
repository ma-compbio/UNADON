import numpy as np
from utils import *
import math
import json
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.nn import TransformerEncoderLayer, TransformerEncoder


torch.autograd.set_detect_anomaly(True)


# Subnetwork for feature extraction
class transformerSep(nn.Module):
    def __init__(self, input_dim, res):
        super().__init__()
        dropout = 0.5
        self.res = res
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, 32))
        self.layers.append(Transpose())
        self.layers.append(nn.BatchNorm1d(32, track_running_stats = False))
        self.layers.append(Transpose())
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(32, 32))
        self.layers.append(Transpose())
        self.layers.append(nn.BatchNorm1d(32, track_running_stats = False))
        self.layers.append(Transpose())
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))

        self.output = nn.Linear(32,32)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = self.output(out)
        return out

# 
class UNADON(nn.Module):
    def __init__(self, input_dim, seq_end, res, num_layer, dim):
        super().__init__()
        self.seqNet = transformerSep(seq_end, res)
        self.histNet = transformerSep(input_dim - seq_end, res)
        self.seq_end = seq_end
        self.domain_clf = DomainClassifier(16)
        self.FC_merge = nn.Linear(64, 64)
        self.FC1 = nn.Linear(64, 16)
        self.scale = 0.01

        self.encoder_layer =  TransformerEncoderLayer(d_model = 64, nhead=8,\
            dim_feedforward = dim, batch_first = True, dropout = 0.3,\
            activation = 'relu')
        self.transformer = TransformerEncoder(self.encoder_layer, num_layer)

        if conf['task'] == 'classification':
            if num_classes == 2:
                self.output = nn.Linear(32, 1)
            else:
                self.output = nn.Linear(32, num_classes)
        elif conf['task'] == 'regression':
            self.output = nn.Linear(16, 1)

        self.pe_base = 1 / 10000 ** (torch.arange(0, 64, 2) / 64)

    def forward(self, x):
        seq = x[:,:,:self.seq_end]
        hist = x[:,:,self.seq_end:]
        seq_out, hist_out = self.seqNet(seq), self.histNet(hist)
        out = torch.cat((seq_out, hist_out), dim = -1)
        out = self.FC_merge(out)
        out = out  + self.scale * \
                self.get_positional_encoding(out.shape, out.get_device())
        # out = nn.ReLU()(out)
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

    def get_positional_encoding(self, shape, device):
        pos = torch.arange(shape[1])
        pe1 = torch.sin(torch.einsum('i,j->ij', pos, self.pe_base))
        pe2 = torch.cos(torch.einsum('i,j->ij', pos, self.pe_base))
        pe = torch.cat((pe1[:,:,None],pe2[:,:,None]), axis = -1)
        pe = pe.reshape((pe.shape[0],pe.shape[1] * 2)) 
        pe_all = pe.unsqueeze(0).repeat((shape[0],1,1)) 
        # pe_all = torch.rand((shape[0],shape[1],shape[2]))
        return pe_all.to(device)



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
        self.layers.append(nn.Linear(50, 3))


    def forward(self,x):
        out = grad_reverse(x)
        for layer in self.layers:
            out = layer(out)
        out = nn.Softmax(dim = -1)(out)
        out = out.permute(0, 2, 1)

        return out

















