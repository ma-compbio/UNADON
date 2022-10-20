import wget
import os
import gzip
import numpy as np
import copy
import json
import h5py
import pybedtools
from itertools import product
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score,confusion_matrix,r2_score, explained_variance_score, mean_squared_error
from torch.utils.data import Dataset
from torch import from_numpy
import torch
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pyBigWig
import math
import time



class SeqData(Dataset):
    def __init__(self, X, y, d, coord):
        self.X = X
        self.y = y
        self.d = d
        self.coord = coord
        

    def get_sample_weight(self):
        # Adjust for cell type similarity
        return [1] * int(len(self.X) * 2 / 3) + [1] * int(len(self.X) / 3)
        
    def __getitem__(self, i):
        X = torch.tensor(self.X[i], dtype=torch.float32)
        y = torch.tensor(self.y[i], dtype = torch.float32)
        d = torch.tensor(self.d[i], dtype = torch.float32)

        coord = [str(list(item)).encode('ascii') for item in self.coord[i]]

        return (X, y, d, coord)

    def __len__(self):
        return len(self.y)


class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return torch.transpose(x, -1, -2)


def byte_to_string(s):
    return s.decode("utf-8") 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_bigwig(output_name, y_true, y_pred, coord):
    idx = [i for i in range(len(y_true)) if type(y_true[i]) == np.float32]
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    coord = coord[idx]

    ref_path = 'data/chromatin_features/K562/K562_WT_SON_2020_08_rep3_25kb_hg38.bw'
    output_path = 'result/' + output_name
    bw = pyBigWig.open(ref_path)
    bw_true = pyBigWig.open(output_path + '_true.bw', 'w')
    bw_pred = pyBigWig.open(output_path + '_pred.bw', 'w')

    bw_true.addHeader([(key, bw.chroms()[key]) for key in bw.chroms().keys()])
    bw_pred.addHeader([(key, bw.chroms()[key]) for key in bw.chroms().keys()])

    if len(coord.shape) == 1:
        coord = np.array([eval(row) for row in coord]).astype(str)
        print(coord)
    elif 'chr' not in coord[0,0]:
        coord = coord.astype(int).astype(str)
        a = np.array(['chr%d' % int(float(i)) for i in coord[:,0]])
        coord[:,0] = a

    coord = list(np.hstack((coord, np.arange(len(coord)).reshape(-1,1))))

    coord = np.array(sorted(coord,key = sort_chrom))
    sort_idx = coord[:,-1].astype(int)
    

    y_true = y_true[sort_idx]
    y_pred = y_pred[sort_idx]


    bw_true.addEntries(coord[:,0], coord[:,1].astype(int),ends = coord[:, 2].astype(int), values = y_true.astype(float))
    bw_pred.addEntries(coord[:,0], coord[:,1].astype(int),ends = coord[:, 2].astype(int), values = y_pred.astype(float))

    print('Finished exporting to BigWig file')



# Load input data and add context for a specific cell type
def load_data_cell(path, chr_list,feature_list,w,histone):
    X = []
    y = []
    with open('config_feat.json', 'r') as f:
        conf = json.load(f)
    coord_list = []
    with h5py.File(path,'r',libver="latest", swmr=True) as f:
        for chr in chr_list:
            print(chr)
            feat = []
            if 'Freq' in feature_list:
                feat.append(np.array(f[chr]['K-mer']['Raw']))
            if 'Freq_pca' in feature_list:
                seq = np.array(f[chr]['K-mer']['PCA'])[:,:20]
                feat.append(seq)
            if 'Raw_seq' in feature_list:
                seq = np.array(f[chr]['Sequence'], dtype = np.int8)
                feat.append(seq)
            # print(np.min(seq, axis = 0))
            if 'Histone' in feature_list:
                names = np.array(f[chr]['Histone']['Name'])
                index = np.array([i for i in range(len(names)) if byte_to_string(names[i])  in histone])
                print(names)
                hist = np.array(f[chr]['Histone']['w21_quantile_transformed'])[:,index]
                feat.append(hist)
            if 'Histone_peak' in feature_list:
                names = np.array(f[chr]['Histone']['Peak_name']).astype(str)
                index = np.array([i for i in range(len(names)) if names[i] in histone])
                hist = np.array(f[chr]['Histone']['Peak_normalized'])[:,index]
                feat.append(hist)
            if 'Motif_svd' in feature_list:
                hist = np.array(f[chr]['Motif_freq_svd'])
                feat.append(hist[:,:])
            feat = np.hstack(feat) 


            index = np.array(f[chr]['TSA-seq'][conf['y']]['Index'])
            feat = feat[index]
            labels = np.array(f[chr]['TSA-seq'][conf['y']]['Values_scale'])
            coord = np.array(f[chr]['Coordinate'])[index]
            print(len(index), len(labels))

            # For w with overlap
            if 'overlap' in path:
                for i in range(0, len(feat) - w * 10):
                    if int(coord[i + w * 10, 1]) - int(coord[i, 1]) <= (w+5) * 25000:
                        idx = np.arange(i, i + w * 10, 10)
                        X.append(feat[idx])
                        y.append(labels[idx])
                        coord_list.append(coord[idx])
            else:
                for i in range(0, len(feat) - w):
                    if int(coord[i+w, 1]) - int(coord[i, 1]) <= (w+5) * 25000:
                        X.append(feat[i: (i+w)])
                        y.append(labels[i: (i+w)])
                        coord_list.append(coord[i: (i+w)])
    if 'overlap' in path:
        ind = np.random.choice(np.arange(len(y)), size = int(len(y)))
    else:
        ind = np.arange(len(y))

    return (np.array(X, dtype = np.float16)[ind], 
            np.array(y, dtype='float')[ind], 
            np.array(coord_list)[ind])


# Load input data
def load_data(cell_type, chr, data_path, feature_list,w, histone = None):
    X, y, d, coord = [],[],[],[]

    for i in range(len(cell_type)):
        cell = cell_type[i]
        (X_c, y_c, coord_c) = load_data_cell(data_path.replace('cell', cell), chr, feature_list, w, histone)
        print(cell, X_c.shape, y_c.shape, coord_c.shape)
        X.append(X_c)
        y.append(y_c)
        d.append(np.ones(y_c.shape) * i)
        coord.append(coord_c[:,:,:3])
        # print(X_c.shape, y_c.shape, coord_c.shape)
    X = np.concatenate(X)
    y = np.concatenate(y)
    d = np.concatenate(d)
    coord = np.concatenate(coord)

    return X, y, d, coord


def merge_pred(y_true_list, y_prob_list, coord_list, epoch,output_name,state):
    print('Calculating merged results')

    merged_true = []
    merged_prob = []
    merged_coord = []

    D = dict()

    for id in np.unique(coord_list):
        D[id] = [None,[]]

    for i in range(len(y_true_list)):

        id = coord_list[i]
        if D[id][0] == None:
            D[id][0] = y_true_list[i]
        else:
            assert(D[id][0] == y_true_list[i])
            pass
        D[id][1].append(y_prob_list[i])


    for id in D.keys():
        if conf['task'] == 'classification':
            pred = np.argmax(D[id][1], axis = 1)
            most_freq = np.argmax(np.bincount(pred))
            vec = [0] * num_classes
            vec[most_freq] = 1
        elif conf['task'] == 'regression':
            # print(prob)
            vec = np.mean(D[id][1])
        if D[id][0] == None:
            print(D[id], id)
        merged_true.append(D[id][0])
        merged_prob.append(vec)
        merged_coord.append(id)

    merged_true = np.array(merged_true)
    merged_prob = np.array(merged_prob)
    merged_coord = np.array(merged_coord)


    print(merged_true.shape, merged_prob.shape, merged_coord.shape)

    if conf['task'] == 'regression':
        idx = [i for i in range(len(merged_coord)) if int(eval(merged_coord[i])[1]) % 25000 == 0]
        print(len(idx),np.array(idx))
        export_bigwig(output_name, merged_true[idx], merged_prob[idx], merged_coord[idx])
        neptune.log_artifact('result/' + output_name + '_true.bw','track/%s_%s_true_%d.bw' % (output_name,state,epoch))
        neptune.log_artifact('result/' + output_name + '_pred.bw','track/%s_%s_pred_%d.bw' % (output_name,state,epoch))

        loss = F.mse_loss(torch.Tensor(merged_prob),torch.Tensor(merged_true))
        print('Merged results')
        reg_report(merged_true, merged_prob)
        return loss

    elif conf['task'] == 'classification':
        (metrics, conf_mat) = clf_report_multiclass(merged_true, merged_prob, store_result = True, coord = merged_coord, name = output_name)
        neptune.log_artifact('result/' + output_name + '_true.bb','track/%s_%s_true_%d.bb' % (output_name,state,epoch))
        neptune.log_artifact('result/' + output_name + '_pred.bb','track/%s_%s_pred_%d.bb' % (output_name,state,epoch))

        return metrics, conf_mat


def reg_report(y_true, y_pred, verbose = True):
    MSE = mean_squared_error(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)
    explained = explained_variance_score(y_true,y_pred)
    if verbose:
        print('MSE', MSE)
        print('Pearson correlation:', pearson)
        print('Spearman:', spearman)
        print('R square:', r2)
        print('Explained variance:', explained)
    return (MSE, pearson, spearman, r2, explained)

chr_dict = dict()
for chr in range(1,10):
    chr_dict['chr%d' % chr] = chr 
for chr in range(10, 23):
    chr_dict['chr%d' % chr] = (chr // 10) + chr * 0.01

def sort_chrom(row):
    return chr_dict[str(row[0])] * 10 ** 12 + int(row[1])



