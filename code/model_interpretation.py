import numpy as np
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
import scipy
import os
from utils import *
import captum
# from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients, Saliency
# from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from captum.attr import IntegratedGradients
import scipy
import seaborn as sns
from transformer_with_attn import TransformerEncoderLayer, TransformerEncoder
from UNADON_model import PrepNet, DomainClassifier





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
        # Compute all targets at the same time
        out = torch.sum(out, dim = 1).reshape(-1,1)
        return out


# Export importance score to bigwig
def export_bigwig(output_path, y, coord, feature_name):

    idx = [i for i in range(len(y)) if type(y_true[i]) == np.float64]
    y = y[idx]
    coord = coord[idx]

    # A reference bigwig file for creating headers. Can use one of the TSA-seq bw
    ref_path = '../../data/chromatin_features/K562/K562_WT_SON_2020_08_rep3_25kb_hg38.bw'
    bw = pyBigWig.open(ref_path)

    bw_new = pyBigWig.open(output_path + '_transformer_IG_%s.bw' % feature_name, 'w')

    bw_new.addHeader([(key, bw.chroms()[key]) for key in bw.chroms().keys()])

    print(coord)

    coord = list(np.hstack((coord, np.arange(len(coord)).reshape(-1,1))))

    coord = np.array(sorted(coord,key = sort_chrom))
    sort_idx = coord[:,-1].astype(int)

    y = y[sort_idx]

    bw_new.addEntries(coord[:,0], coord[:,1].astype(int),ends = coord[:, 2].astype(int), values = y.astype(float))
    bw_new.close()

    # print('Finished exporting to BigWig file')


def model_interpretation(conf):

    id = get_free_gpu()
    device = torch.device("cuda:%d" % id) 

    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    (X_test, y_test, d_test, coord_test) = load_data(conf['IML_cell_type'], \
        conf['testing_chr'], conf['test_data_path'], conf['feature'], \
        conf['window_size'], conf['y'], conf['histone'])

    dts_test = SeqData(X_test, y_test, d_test, coord_test)

    loader_test = DataLoader(dataset = dts_test, 
                              batch_size = conf['batch'],
                              pin_memory=True,
                              num_workers = 2,
                              shuffle = False)


    input_dim = X_train.shape[-1]

    model = UNADON(input_dim, 20, conf['dense_dim'], conf['dense_num_layers'],\
    conf['nhead'],  conf['attn_hidden_dim'], conf['attn_layers'], conf["dropout"])
    model.to(device)


    model_path = conf['output_path'] + 'model/%s_%s_%s_best.pth' % (conf['mode'],conf['y'],conf['IML_cell_type'])

    # Load the best model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.train()
    iml = IntegratedGradients(model)

    data = []
    row = []
    col = []
    batch_size = 8 # Big memory usage

    X = []
    coord = []
    
    for i in range(len(X_test)):
        (x, y, d, c) = dts_test.__getitem__(i)
        X.append(x.numpy())
        coord.append(c)
    
    X = np.array(X)
    coord = np.array(coord)

    print(len(coord), len(X))

    D = dict()

    for id in np.unique(coord):
        D[id] = []


    for i in range(0,len(X), batch_size):
        print(i)
        start, end = i, min(len(X), i + batch_size)
        X_batch = torch.Tensor(X[start:end]).to(device)
        coord_batch = coord[start:end]
        imp_loc = np.zeros((batch_size, X.shape[-2], X.shape[-1]))
        if end - start != batch_size:
            continue
        imp_loc += np.absolute(iml.attribute(X_batch, torch.zeros((1,X_batch.shape[1],X_batch.shape[2])).to(device)).detach().cpu().numpy())      
        for j in range(imp_loc.shape[0]):
            for k in range(imp_loc.shape[1]):
                D[coord_batch[j][k]].append(imp_loc[j,k])
        del X_batch


    imp_list = []
    coord_list = []

    for id in D.keys():
        if len(D[id]) != 0:
            imp_list.append(np.average(D[id], axis = 0))
        else:
            imp_list.append(np.zeros((X.shape[-1])))
        coord_list.append(id)

    imp_list = np.array(imp_list)
    coord_list = np.array(coord_list)

    print(imp_list.shape)
    print(coord_list.shape)
    print(coord_list[:10])


    np.savetxt(conf['output_path'] + '%s_%s_%s_transformer_IG_values.txt' % (conf['mode'], conf['IML_cell_type'], conf['y']), np.array(imp_list))
    np.savetxt(conf['output_path'] + '%s_%s_%s_transformer_IG_coordinate.txt' % (conf['mode'], conf['IML_cell_type'], conf['y']), np.array(coord_list), fmt = '%s')



    importance = np.loadtxt(conf['output_path'] + '%s_%s_%s_transformer_IG_values.txt' % (conf['mode'], conf['IML_cell_type'], conf['y']))
    coord = np.loadtxt(conf['output_path'] + '%s_%s_%s_transformer_IG_coordinate.txt' % (conf['mode'], conf['IML_cell_type'], conf['y']), dtype = str)
    # print(coord)
    coord_new = []
    for row in coord:
        coord_new.append([row[0][5:-2], row[1][2:-2], row[2][2:-3]])

    coord = np.array(coord_new)
    print(coord.shape)

    # Export the sequence importance score
    export_bigwig(conf['output_path'] + '%s_%s_%s' % (conf['mode'], conf['IML_cell_type'], conf['y']), np.sum(importance[:,:20], axis = 1).astype(float), coord, 'seq')


    for i in range(len(conf['histone'])):
        export_bigwig(conf['output_path'] + '%s_%s_%s' % (conf['mode'], conf['IML_cell_type'], conf['y']), importance[:,20+i].astype(float), coord, conf['histone'][i])

    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    with open('../config.json', 'r') as f:
        conf = json.load(f)


    model_interpretation(conf)

