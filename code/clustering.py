from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
import numpy as np
import pandas as pd
from utils import *
import pybedtools
import matplotlib.pyplot as plt
import matplotlib.style as style
import pyBigWig
import seaborn as sns
from sklearn.preprocessing import quantile_transform, QuantileTransformer, MinMaxScaler


# Clustering of the feature contribution patterns
def clustering():
    cell_types = ['K562','H1','HCT116','HFFc6']
    tsa_type = 'LMNB'
    mode = 'Single'

    histones = ["ATAC-seq","H2A.Z","H3K4me1","H3K4me2", "H3K4me3", "H3K9me3","H3K27me3", "H3K27ac", "H3K36me3"]

    record = [['mode', 'Sequence'] + histones]

    contri_all = []
    coord_all = []


    # Load the sequence and epigenomics importance scores for all cell types
    for mode in ['Single']:
        for tsa_type in ['LMNB']:
            for cell_type in ['K562','H1','HCT116','HFFc6']:
                seq_path = 'result/%s_%s_%s_transformer_IG_seq.bw' % (mode, cell_type, tsa_type)
                bw_seq = pyBigWig.open(seq_path, 'r')

                coord = []

                plt.figure(figsize = (20,3))
                all_contri = []
                all_contri_seq = []

                for i in range(2,23,2):
                    coord.append(np.array([['chr%d' % i, x[0], x[1]] for x in bw_seq.intervals('chr%d' % i)]))
                    seq_contri = np.array([float(x[2])  for x in bw_seq.intervals('chr%d' % i)])

                    total_contri = copy.copy(seq_contri)
                    chr_contri = [seq_contri]
                    for hist in histones:
                        bw_hist = pyBigWig.open('result/%s_%s_%s_transformer_IG_%s.bw' % (mode, cell_type, tsa_type, hist), 'r')
                        start = np.array([int(x[0])  for x in bw_hist.intervals('chr%d' % i)])
                        hist_contri = np.array([float(x[2])  for x in bw_hist.intervals('chr%d' % i)])
                        total_contri += hist_contri
                        chr_contri.append(hist_contri)
                    chr_contri = np.array(chr_contri)
                    all_contri.append(chr_contri)
                    all_contri_seq.append(seq_contri)
            #         print(np.mean(all_contri / total_contri, axis = 1))


                all_contri = np.hstack(all_contri)
                all_contri_seq = np.concatenate(all_contri_seq)
                coord = np.concatenate(coord)
                print(coord.shape)
                coord = np.hstack((coord, np.array([cell_type] * len(coord)).reshape(-1,1)))
                coord_all.append(coord)
                print(all_contri.shape)
                contri = all_contri.T
                print(np.average(contri, axis = 0))
                for i in range(10):
                    nonzero_ind = np.nonzero(contri[:,i] != 0)[0]
                #     nonzero_ind = np.arange(len(contri))
                    print(len(nonzero_ind))
                    contri[:,i][nonzero_ind] = quantile_transform(contri[:,i][nonzero_ind].reshape(-1,1), n_quantiles = 10000).reshape(-1)

                contri_all.append(contri)

    contri_all = np.concatenate(contri_all)
    coord_all = np.concatenate(coord_all)

    np.random.seed(31415)

    print(contri_all.shape)
    clustering = KMeans(n_clusters=6, random_state=413).fit(contri_all)
    # clustering = AgglomerativeClustering(n_clusters=6).fit(contri)
    cls = clustering.labels_
    # Hard coding - better organize the clusters
    cls_num = [6,1,5,2,3,4]
    for i in range(6):
        print('Cluster %d' % cls_num[i])
        print(i, np.count_nonzero(cls == i), np.count_nonzero(cls == i) / len(cls))
        ind = np.nonzero(cls == i)[0]
        print(ind[:50])
        print(np.mean(contri_all[ind], axis = 0))
        x = contri_all[ind].reshape(-1,)
        y = (['Sequence'] + histones) * len(ind)
        
        plt.figure(figsize = (3,5))
        sns.boxplot(x=x, y=y, fliersize = 0, palette = ['#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2'])
        plt.xlim(-0.1,1.1)
        
        plt.savefig('result/cluster_%d.pdf' % cls_num[i])
        np.savetxt('result/clustering_%d.bed' % cls_num[i], coord_all[ind], delimiter = '\t', fmt = '%s')    

    cls = np.array(['Cluster_%d' % c  for c in cls])