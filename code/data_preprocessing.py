import pyBigWig
import pybedtools
import numpy as np
import h5py
import time
from Bio import SeqIO
import h5py
from itertools import product
import os
from sklearn.decomposition import PCA
import json

'''
The data preprocessing pipeline will generate a hdf5 file with the following structure:
processed_data.hdf5
    - Chr i
        - Sequence 
        - Coord
        - K-mer 
            - k 
                - raw
                - name
                - Mappable base
            - pca
        - Histone
            - Peak
            - Peak_normalized 
            - Peak_name
            - Names
        - TSA-seq
            - type (SON, LMNB, ..)
                - Values_scale
                - Index
'''



# Count k-mer frequency for a given sequence
def kmer_counter(L, k, return_key = False):
    '''
    Get k-mers
    A: 1, C: 2, G: 3, T: 4
    '''
    nt = {'1': 'A', '2': 'C', '3': 'G', '4': 'T'}
    L = [str(int(i)) for i in L]
    freq = dict()
    kmers = product('1234', repeat = k)
    kmers = ["".join(item) for item in kmers]
    for kmer in kmers:
        freq[kmer] = 0
    zero_counter = 0
    
    
    for i in range(len(L)-k+1):
        try:
            freq[''.join(L[i:(i+k)])] += 1
        except:
            # print('0 encountered!')
            zero_counter += 1
            pass
    freq_merged = dict()
    for kmer in freq.keys():
        complement = str(int('5'*len(kmer)) - int(kmer[::-1])) # Reverse complement
        if complement not in freq_merged.keys():
            freq_merged[kmer] = freq[kmer]
        else:
            freq_merged[complement] += freq[kmer]
    for kmer in freq_merged.keys():
        freq_merged[kmer] /= (len(L) - len(kmer) + 1)
    
    freq_list = [freq_merged[i] for i in sorted(list(freq_merged.keys()))]
    if return_key:
        keys = sorted(list(freq_merged.keys()))
        nt_keys = []
        for kmer in keys:
            mer_nt = ''.join([nt[i] for i in kmer])
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
def kmer_from_anno(conf):
    res = conf['resolution']
    seq_path = conf['seq_path']
    output_file = conf['output_file'].replace('cell',conf['cell_type'])
    # Create the output h5 file
    if os.path.isfile(output_file):
        os.remove(output_file)
    output_file = h5py.File(output_file,'w')
    for i in range(1,23):
        t = time.time()
        chr = 'chr%d' % i
        for rec in SeqIO.parse(seq_path + chr + '.fa', 'fasta'):
            rec = rec
        # print(chr, len(rec))
        # Define from sequence
        anno_mat = [[chr, start, start + res] for start in range(0,len(rec) - res, res)]
        anno = pybedtools.BedTool(anno_mat)
        anno = np.array(anno)
        for row in anno:
            if int(row[2]) - int(row[1]) != res:
                row[2] = str(int(row[1]) + res)
        anno_mat = np.array([[str(t[0]).encode('ascii'),\
                              str(t[1]).encode('ascii'),\
                              str(t[2]).encode('ascii')] for t in anno], dtype = object)
        # print(len(anno_mat))
        # print(anno_mat[:2])
        anno = anno.tolist()
        k_list = conf['kmer_list']

        freq_mat_all = {}
        for k in k_list:
            freq_mat_all[k] = []
            
        for i in range(0,len(anno), conf['resolution']):
            anno_batch = pybedtools.BedTool(anno[i:(i+conf['resolution'])])
            print(len(anno_batch))
            anno_batch = anno_batch.sequence(fi = seq_path + chr + '.fa', s = True)
            fasta_sequences = SeqIO.parse(open(anno_batch.seqfn),'fasta')
            seq = np.array([genome_to_num(str(fasta.seq)) for fasta in fasta_sequences])
            num_mappable = np.array([count_mappable(seq_array) for seq_array in seq])
    #         print(num_mappable.tolist())
            print('Sequence shape:', seq.shape)
            for k in k_list:
                freq_mat = [kmer_counter(data, k) for data in seq]
                freq_mat = np.array(freq_mat)
                freq_mat_all[k].append(freq_mat)
                print(k, freq_mat.shape)
        
        for k in k_list:
            freq_mat_all[k] = np.concatenate(freq_mat_all[k])
        g = output_file.create_group(chr)
#         g.create_dataset('Sequence', data = seq, dtype = np.int8) # Save raw sequence
        h = g.create_group('K-mer')
        for k in k_list:
            kmer_group = h.create_group(str(k))
            kmer_group.create_dataset('Raw', data = freq_mat_all[k])
            dt = h5py.special_dtype(vlen=str)
            kmer_group.create_dataset('Mappable_base', data = num_mappable)
            kmer_group.create_dataset('Name', data = kmer_counter([], k, True))
        g.create_dataset('Coordinate', data = anno_mat, dtype = dt)
            
        print(chr,freq_mat.shape)
        print('Time spent:', time.time() - t)

    output_file.close()

# PCA transformation for k-mer frequency
def kmer_PCA(conf):
    np.random.seed(42) # Make sure that the embedding is reproducible
    k_list = conf['kmer_list']
    f = h5py.File(conf['output_file'].replace('cell',conf['cell_type']), 'a')
    kmer_all = []
    for k in k_list:
        np.random.seed(42)
        kmer_mat = []
        # Only use the odd-numbered chromosomes to avoid data leakage
        for i in range(1,23,2): 
            print(i)
            chr = 'chr%d' % i
            kmer_mat.append(np.array(f[chr]['K-mer'][str(k)]['Raw']))
        kmer_mat = np.concatenate(kmer_mat)
        kmer_all.append(kmer_mat)
    kmer_all = np.hstack(kmer_all)
    pca = PCA(n_components = 20)
    kmer_pca = pca.fit_transform(kmer_all)
    print('PCA', kmer_mat.shape, kmer_pca.shape, pca.explained_variance_ratio_.cumsum())
    
    for i in range(1,23):
        print(i)
        chr = 'chr%d' % i
        if 'PCA' in f[chr]['K-mer'].keys():
            del f[chr]['K-mer']['PCA']
        kmer_chr = []
        for k in k_list:
            kmer_chr.append(np.array(f[chr]['K-mer'][str(k)]['Raw']))
        kmer_chr = np.hstack(kmer_chr)
        reduced = pca.transform(kmer_chr)
        f[chr]['K-mer'].create_dataset('PCA', data = reduced) 
        print(np.array(f[chr]['K-mer']['PCA']).shape)
    f.close()

# Count the occurrences of ATAC-seq and histone signal peaks
def histone_peak_processing(conf):
    cell_type = conf['cell_type']
    f = h5py.File(conf['output_file'].replace('cell', cell_type), 'a')
    # print(f.keys())
    for i in range(1,23):
        print(i)
        chrom = 'chr%d' % i
        chr_anno = list(f[chrom]['Coordinate'])
        chr_anno = pybedtools.BedTool(chr_anno)
        count_list = []
        for mark in conf['epi_name']:
            print(mark)
            # Allow the processing of histone data with different format
            peak_path = conf['signal_path'] + '%s/%s_%s.bed.gz' % (cell_type, cell_type, mark)
            if not os.path.exists(peak_path):
                peak_path = conf['signal_path'] + '%s/%s_%s.bed' % (cell_type, cell_type, mark)
            if not os.path.exists(peak_path):
                peak_path = conf['signal_path'] + '%s/CUT_RUN/%s_%s.CUT_RUN.bed' % (cell_type, cell_type, mark)
            peak_anno = pybedtools.BedTool(peak_path)
    #         print('Before merging:',len(peak_anno))
            peak_anno = peak_anno.sort()
            peak_anno = peak_anno.merge(d = -1)
    #         print('After merging:',len(peak_anno))
            
            intersect = chr_anno.intersect(peak_anno, wao = True)
            intersect = intersect.merge(d = -conf['resolution'] + 1, c = 7, o = 'sum')
            intersect = np.array(intersect)
    #         print(len(intersect))
    #         print(len(np.array(f[chrom]['Coordinate'])))
            
            assert(np.array_equal(intersect[:,:-1].astype(str), np.array(f[chrom]['Coordinate']).astype(str)))
            count_list.append(intersect[:,-1].astype(float) / conf['resolution'])
            
        if 'Histone' not in f[chrom].keys():
            f[chrom].create_group('Histone')
        if 'Peak' in f[chrom]['Histone'].keys():
            del f[chrom]['Histone']['Peak']
        if 'Peak_name' in f[chrom]['Histone'].keys():
            del f[chrom]['Histone']['Peak_name']
        count_list = np.vstack(count_list).T
        # print(count_list.shape)
        f[chrom]['Histone'].create_dataset('Peak', data = count_list)
        dt = h5py.special_dtype(vlen=str)
        f[chrom]['Histone'].create_dataset('Peak_name', data = [str(mark).encode('ascii') for mark in conf['epi_name']])
        
        
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

# Load and scale TSA-seq signal to between -1 and 1
def tsa_seq_preprocessing(conf):
    centromere_coord = np.loadtxt(conf['centromere_path'], dtype = str, delimiter = '\t')
    f = h5py.File(conf['output_file'].replace('cell',conf['cell_type']), 'a')
    cell_type = conf['cell_type']

    for tsa_type in conf['tsa_type']: # the target nuclear bodies
        print(tsa_type)
        bw = pyBigWig.open(conf['signal_path'] + '%s/%s_%s_TSA_seq.bw' % (cell_type, cell_type, tsa_type))
        # print(bw.chroms())
        # print(bw.header())
        anno = []
        hanning = np.hanning(21)

        scaling_factor = conf['tsa_scaling'][cell_type][tsa_type]
        
        # Smoothing
        for j in [1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,3,4,5,6,7,8,9]:
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
            values[values <= 0] = values[values <= 0] / scaling_factor[0]
            values[values > 0] = values[values > 0] / scaling_factor[1]
            values = np.clip(values, -1, 1)
            smoothed_values = []
            for i in range(len(values)):
                idx = np.arange(max(0, (i - 10)) , min(len(values) , i + 11))
                v = np.dot(np.array(values[idx]),np.array(hanning[(idx - i) + 10])) / np.sum(hanning[(idx - i) + 10])
                smoothed_values.append(v)
            chr_anno = np.hstack((chr_anno, np.array(smoothed_values).reshape(-1,1)))
            
            
            for i in range(len(chr_anno) - 1):
                # Remove bins with too many unmapped sequences
                k = list(f[chr]['K-mer'].keys())[0]
                kmer_prop = np.array(f[chr]['K-mer'][k]['Raw'][i])
                # print(kmer_prop)
                if np.sum(kmer_prop) <= 0.8:
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
        # print(anno[:,4])
        
        # print(list(anno[:20]))
        # print(anno.shape)
        for j in range(1,23):
            chr = 'chr%d' % j
            chr_anno = anno[np.nonzero((anno[:,0] == chr))]
            if 'Values_scale' in f[chr]['TSA-seq'][tsa_type].keys():
                del f[chr]['TSA-seq'][tsa_type]['Values_scale']
            f[chr]['TSA-seq'][tsa_type].create_dataset('Values_scale', data = chr_anno[:,4].astype(float))
        
        # Quantile transformation was tested but not used in the final version
        # anno[:,4] = quantile_transform(anno[:,4].reshape(-1,1).astype(float), \
        #     output_distribution = 'uniform', n_quantiles = 1000000, subsample = 1000000).reshape(-1,) * 2 - 1
        # for j in range(1,23):
        #     chr = 'chr%d' % j
        #     chr_anno = anno[np.nonzero((anno[:,0] == chr))]
        #     if 'Values_quantile' in f[chr]['TSA-seq'][tsa_type].keys():
        #         del f[chr]['TSA-seq'][tsa_type]['Values_quantile']
        #     f[chr]['TSA-seq'][tsa_type].create_dataset('Values_quantile', data = chr_anno[:,4].astype(float))
    f.close()

def main(conf):

    # Step 1: Process sequnece features and create the hdf5 file
    print('Processing sequence features...')
    kmer_from_anno(conf)

    # Step 2: Process the epigenomic features
    print('Processing epigenomic features...')
    histone_peak_processing(conf)

    # Step 3: Process the TSA-seq signals
    print('Processing TSA-seq signals...')
    tsa_seq_preprocessing(conf)


if __name__ == '__main__':
    with open('../config/config_data.json', 'r') as f:
        conf = json.load(f)
    main(conf)