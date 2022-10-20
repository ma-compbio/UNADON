import pyBigWig
import pybedtools
import numpy as np
import h5py
import pandas as pd
import time
from Bio import SeqIO
import h5py
from itertools import product
import os
from sklearn.decomposition import PCA

# Count k-mer frequency for a given sequence
def kmer_counter(L, return_key = False):
    '''
    Get 5-mers and 6-mers
    A: 1, C: 2, G: 3, T: 4
    '''
    nt = {'1': 'A', '2': 'C', '3': 'G', '4': 'T'}
    L = [str(int(i)) for i in L]
    freq = dict()
    mers_6 = product('1234', repeat = 5)
    mers_6 = ["".join(item) for item in mers_6]
    for kmer in mers_6:
        freq[kmer] = 0
    mers_7 = product('1234', repeat = 6)
    mers_7 = ["".join(item) for item in mers_7]
    for kmer in mers_7:
        freq[kmer] = 0
    zero_counter = 0
    for i in range(len(L)-4):
        try:
            freq[''.join(L[i:i+5])] += 1
        except:
            # print('0 encountered!')
            zero_counter += 1
            pass
    for i in range(len(L)-5):
        try:
            freq[''.join(L[i:i+6])] += 1
        except:
            # print('0 encountered!')
            zero_counter += 1
    freq_merged = dict()
    # merge reverse complementary k-mers
    index = 0
    for mer in freq.keys():
        complement = str(int('5'*len(mer)) - int(mer[::-1]))
#         print(mer,complement)
        if complement not in freq_merged.keys():
            freq_merged[mer] = freq[mer]
        else:
            freq_merged[complement] += freq[mer]
    for mer in freq_merged.keys():
        freq_merged[mer] /= (len(L) - len(mer) + 1)

    # print(len(freq.keys()), len(freq_merged.keys()))
    freq_list = [freq_merged[i] for i in sorted(list(freq_merged.keys()))]
    if return_key:
        keys = sorted(list(freq_merged.keys()))
        nt_keys = []
        for mer in keys:
            mer_nt = ''.join([nt[i] for i in mer])
            nt_keys.append(mer_nt.encode('ascii'))
        return (freq_list, nt_keys)
    else:
        return freq_list

# Covnert ACGT to numbers
def genome_to_num(seq):
    seq = list(seq.upper())
    D = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
    mat = [D[i] for i in seq]
    return mat

# Count the number of mappable bases
def count_mappable(seq):
    return len(seq) - len(np.where(seq == 0)[0])


# Compute k-mer frequency and store in hdf5 file
def kmer_from_anno(cell_type):
    res = 25000
    seq_path = 'data/sequence/'
    output_path = 'data/processed_data/hg38_%s_25kb_fixed_kmer.hdf5' % cell_type
    if os.path.isfile(output_path):
        os.remove(output_path)
    output_file = h5py.File(output_path,'w')
    for i in range(1,23):
        t = time.time()
        chr = 'chr%d' % i
        for rec in SeqIO.parse(seq_path + chr + '.fa', 'fasta'):
            rec = rec
        print(chr, len(rec))
        # define from sequence
        anno_mat = [[chr, start, start + res] for start in range(0,len(rec) - res, 25000)]
        anno = pybedtools.BedTool(anno_mat)
        anno = np.array(anno)
        for row in anno:
            if int(row[2]) - int(row[1]) != res:
                row[2] = str(int(row[1]) + res)
        anno_mat = np.array([[str(t[0]).encode('ascii'),\
                              str(t[1]).encode('ascii'),\
                              str(t[2]).encode('ascii')] for t in anno], dtype = object)
        print(len(anno_mat))
        print(anno_mat[:2])
        anno = anno.tolist()
        
        freq_mat_all = []
        for i in range(0,len(anno),25000):
            anno_batch = pybedtools.BedTool(anno[i:(i+25000)])
            anno_batch = anno_batch.sequence(fi = seq_path + chr + '.fa', s = True)
            fasta_sequences = SeqIO.parse(open(anno_batch.seqfn),'fasta')
            seq = np.array([genome_to_num(str(fasta.seq)) for fasta in fasta_sequences])
            num_mappable = np.array([count_mappable(seq_array) for seq_array in seq])
    #         print(num_mappable.tolist())
#             print('Sequence shape:', seq.shape)
            freq_mat = [kmer_counter(data) for data in seq]
            freq_mat = np.array(freq_mat)
            freq_mat_all.append(freq_mat)
        freq_mat = np.concatenate(freq_mat_all)
        g = output_file.create_group(chr)
#         g.create_dataset('Sequence', data = seq, dtype = np.int8)
        h = g.create_group('K-mer')
        h.create_dataset('Raw', data = freq_mat)
        dt = h5py.special_dtype(vlen=str)
        h.create_dataset('Mappable_base', data = num_mappable)
        h.create_dataset('Name', data = kmer_counter([],True))
        g.create_dataset('Coordinate', data = anno_mat, dtype = dt)
        print(chr,freq_mat.shape)
        print('Time spent:', time.time() - t)

    output_file.close()


# PCA transformation for k-mer frequency
def kmer_PCA(path):
    f = h5py.File(path,'a')
    kmer_mat = []
    # Only fit on the odd chromosomes to avoid overfitting
    for i in range(1,23,2):
        print(i)
        chr = 'chr%d' % i
        kmer_mat.append(np.array(f[chr]['K-mer']['Raw']))
        
    kmer_mat = np.concatenate(kmer_mat)
    print(kmer_mat.shape)

    pca = PCA(n_components = 50)
    kmer_pca = pca.fit_transform(kmer_mat)
    print(kmer_pca.shape,pca.explained_variance_ratio_.cumsum())


    for i in range(1,23):
        print(i)
        chr = 'chr%d' % i
        reduced = pca.transform(np.array(f[chr]['K-mer']['Raw']))
        if 'PCA' in f[chr]['K-mer'].keys():
            del f[chr]['K-mer']['PCA']
        f[chr]['K-mer'].create_dataset('PCA', data = reduced) 
        print(np.array(f[chr]['K-mer']['Raw']).shape,np.array(f[chr]['K-mer']['PCA']).shape)


# Count the occurrences of ATAC-seq and histone signal peaks
def histone_peak_processing(cell_type, output_path, markers):
    f = h5py.File(output_path,'a')
    print(f.keys())
    for i in range(1,23):
        print(i)
        chrom = 'chr%d' % i
        chr_anno = list(f[chrom]['Coordinate'])
        chr_anno = pybedtools.BedTool(chr_anno)
        count_list = []
        for mark in markers:
            print(mark)
            peak_path = 'data/chromatin_features/%s/%s_%s.bed.gz' % (cell_type, cell_type, mark)
            if not os.path.exists(peak_path):
                peak_path = 'data/chromatin_features/%s/%s_%s.bed' % (cell_type, cell_type, mark)
            if not os.path.exists(peak_path):
                peak_path = 'data/chromatin_features/%s/CUT_RUN/%s_%s.CUT_RUN.bed' % (cell_type, cell_type, mark)
            peak_anno = pybedtools.BedTool(peak_path)
    #         print('Before merging:',len(peak_anno))
            peak_anno = peak_anno.sort()
            peak_anno = peak_anno.merge(d = -1)
    #         print('After merging:',len(peak_anno))
            
            intersect = chr_anno.intersect(peak_anno, wao = True)
            intersect = intersect.merge(d = -24999, c = 7, o = 'sum')
            intersect = np.array(intersect)
    #         print(len(intersect))
    #         print(len(np.array(f[chrom]['Coordinate'])))
            
            assert(np.array_equal(intersect[:,:-1].astype(str), np.array(f[chrom]['Coordinate']).astype(str)))
            count_list.append(intersect[:,-1].astype(float) / 25000)
            
        if 'Histone' not in f[chrom].keys():
            f[chrom].create_group('Histone')
        if 'Peak' in f[chrom]['Histone'].keys():
            del f[chrom]['Histone']['Peak']
        if 'Peak_name' in f[chrom]['Histone'].keys():
            del f[chrom]['Histone']['Peak_name']
        count_list = np.vstack(count_list).T
        print(count_list.shape)
        f[chrom]['Histone'].create_dataset('Peak', data = count_list)
        dt = h5py.special_dtype(vlen=str)
        f[chrom]['Histone'].create_dataset('Peak_name', data = [str(mark).encode('ascii') for mark in markers])
        
        
    data = []

    for i in range(1,23):
        chrom = 'chr%d' % i
        data.append(np.array(f[chrom]['Histone']['Peak']))

    data = np.concatenate(data)
    print(data.shape)
    print(np.array(f[chrom]['Histone']['Peak_name']))
    mean = np.mean(data, axis = 0)
    print(mean)

    coord_list = []
    peak_list = []

    for i in range(1,23):
        chrom = 'chr%d' % i
        peak = np.array(f[chrom]['Histone']['Peak'])
        print(peak.shape)
        if 'Peak_normalized' in f[chrom]['Histone'].keys():
            del f[chrom]['Histone']['Peak_normalized']
        f[chrom]['Histone'].create_dataset('Peak_normalized', data = (peak / (mean + 1e-9)) * 0.01)
        coord_list.append(np.array(f[chrom]['Coordinate']))
        peak_list.append(np.array(f[chrom]['Histone']['Peak_normalized']))

    f.close()

# Scaled TSA-seq signal to between -1 and 1
def tsa_seq_preprocessing(cell_type, path, file_names, scaling_factor):
    centromere_coord = np.loadtxt('data/hg38_centromere.bed', dtype = str, delimiter = '\t')
    f = h5py.File('data/processed_data/hg38_%s_25kb_fixed_kmer.hdf5' % cell_type,'a', swmr = True)

    for tsa_type in file_names.keys():
        print(tsa_type)
        bw = pyBigWig.open(path + file_names[tsa_type])
        print(bw.chroms())
        print(bw.header())
        anno = []
        hanning = np.hanning(21)
        
        # Smoothing
        for j in [1,10,11,12,13,14,15,16,17,18,19,2,21,22,3,4,5,6,7,8,9]:
            chr = 'chr%d' % j
            intv = np.array(f[chr]['Coordinate'])
            row = intv[0]
            print(bw.stats(row[0].decode("utf-8") ,int(row[1]),int(row[2]))[0])
            chr_anno = np.array([[row[0].decode("utf-8"), row[1], row[2], bw.stats(row[0].decode("utf-8") ,int(row[1]),int(row[2]))[0]] for row in intv])
            centromere_chr = centromere_coord[np.nonzero(centromere_coord[:,0] == chr)[0]]
            index = []
            print(chr_anno)
            print(len(chr_anno))
            values = chr_anno[:,-1].astype(float)
            values[values <= 0] = values[values <= 0] / scaling_factor[cell_type][tsa_type][0]
            values[values > 0] = values[values > 0] / scaling_factor[cell_type][tsa_type][1]
            values = np.clip(values, -1, 1)
            smoothed_values = []
            for i in range(len(values)):
                idx = np.arange(max(0, (i - 10)) , min(len(values) , i + 11))
                v = np.dot(np.array(values[idx]),np.array(hanning[(idx - i) + 10])) / np.sum(hanning[(idx - i) + 10])
                smoothed_values.append(v)
            chr_anno = np.hstack((chr_anno, np.array(smoothed_values).reshape(-1,1)))
            
            
            for i in range(len(chr_anno) - 1):
                # Remove bins with too many unmapped sequences
                kmer_prop = np.array(f[chr]['K-mer']['Raw'][i])
                if np.sum(kmer_prop) <= 1.6:
    #                 print(chr,i,'too many Ns')
                    continue
                (start, end) = chr_anno[i,1], chr_anno[i,2]
                # Remove bins in the centromere region 
                is_centromere = False
                for intv in centromere_chr:
                    if int(start) >= int(intv[1]) and int(start) <= int(intv[2]):
    #                     print(chr,i,'Centromere region')
                        is_centromere = True
                        break
                if is_centromere:
                    continue
                index.append(i)
            index = np.array(index)
            
            if 'TSA-seq' not in f[chr].keys():
                f[chr].create_group('TSA-seq')
            if tsa_type in f[chr]['TSA-seq'].keys():
                del f[chr]['TSA-seq'][tsa_type]
            g = f[chr]['TSA-seq'].create_group(tsa_type)
            g.create_dataset('Index', data = index)
                
            print(len(chr_anno), len(index))
            chr_anno = chr_anno[index]
            
            
            anno.extend(chr_anno.tolist())

        anno = np.array(anno)
        print(anno[:,4])
        
        print(list(anno[:20]))
        print(anno.shape)
        for j in range(1,23):
            if j == 20:
                continue
            chr = 'chr%d' % j
            chr_anno = anno[np.nonzero((anno[:,0] == chr))]
            if 'Values_scale' in f[chr]['TSA-seq'][tsa_type].keys():
                del f[chr]['TSA-seq'][tsa_type]['Values_scale']
            f[chr]['TSA-seq'][tsa_type].create_dataset('Values_scale', data = chr_anno[:,4].astype(float))
        
        anno[:,4] = quantile_transform(anno[:,4].reshape(-1,1).astype(float), \
            output_distribution = 'uniform', n_quantiles = 1000000, subsample = 1000000).reshape(-1,) * 2 - 1
        for j in range(1,23):
            if j == 20:
                continue
            chr = 'chr%d' % j
            chr_anno = anno[np.nonzero((anno[:,0] == chr))]
            if 'Values_quantile' in f[chr]['TSA-seq'][tsa_type].keys():
                del f[chr]['TSA-seq'][tsa_type]['Values_quantile']
            f[chr]['TSA-seq'][tsa_type].create_dataset('Values_quantile', data = chr_anno[:,4].astype(float))
    f.close()

    
