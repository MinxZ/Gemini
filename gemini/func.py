"""
Mingxin Zhang
Help functions
"""
import os

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import moment


def out_string_nets(net, org):
    if net == 'GeneMANIA' or 'ex' in net or net == 'human_drug':
        org2binomial = {'yeast': 'Saccharomyces_cerevisiae',
                        'human': 'Homo_sapiens',
                        'human_match': 'Homo_sapiens',
                        'human_drug': 'Homo_sapiens',
                        'mouse': 'Mus_musculus'}

        binomial = org2binomial[org]
        lst = pd.read_csv(f"data/raw/{binomial}/networks.txt",
                          sep='\t').File_Name
        string_nets = []

    if net == 'net12':
        data = pd.read_csv('data/raw/drug/data1.csv')
        seq3 = list(data['data_source(s)'])
        seq3 = [set(item.split(',')) for item in seq3]
        string_nets = list(set([j for i in seq3 for j in i]))
        string_nets = sorted(string_nets)
        string_nets = [n+'_gm' for n in string_nets]
    elif net == 'net12_ex':
        data = pd.read_csv('data/raw/drug/data1.csv')
        seq3 = list(data['data_source(s)'])
        seq3 = [set(item.split(',')) for item in seq3]
        string_nets = list(set([j for i in seq3 for j in i]))
        string_nets = sorted(string_nets)
        string_nets = [n+'_gm' for n in string_nets]
    elif net == 'GeneMANIA':
        for network in lst:
            net_path = f"data/raw/{binomial}/{network}"
            net_name = net_path.split('/')[-1].replace('.txt', '')
            string_nets.append(net_name)
        string_nets = sorted(string_nets)

    elif net == 'GeneMANIA_ex':
        for network in lst:
            net_path = f"data/raw/{binomial}/{network}"
            net_name = net_path.split('/')[-1].replace('.txt', '')
            string_nets.append(net_name)
        string_nets = sorted(string_nets)
        string_nets = [n+'_gm' for n in string_nets]
    elif net == 'human_drug':
        data = pd.read_csv('data/raw/drug/data1.csv')
        seq3 = list(data['data_source(s)'])
        seq3 = [set(item.split(',')) for item in seq3]
        string_nets = list(set([j for i in seq3 for j in i]))
        string_nets_drug = sorted(string_nets)

        string_nets = []
        for network in lst:
            net_path = f"data/raw/{binomial}/{network}"
            net_name = net_path.split('/')[-1].replace('.txt', '')
            string_nets.append(net_name)
        string_nets = sorted(string_nets)
        print(len(string_nets))
        string_nets += string_nets_drug
        print(len(string_nets))
        string_nets = [n+'_gm' for n in string_nets]
    elif net == 'mashup':
        string_nets = ['neighborhood', 'fusion', 'cooccurence',
                       'coexpression',  'experimental', 'database']
        string_nets = sorted(string_nets)
    elif net == 'mashup_ex':
        string_nets = [f'{n}_gm' for n in ['neighborhood', 'fusion',
                                           'cooccurence',
                                           'coexpression',  'experimental',
                                           'database']]
        string_nets = sorted(string_nets)
    elif net == 'mashup_GeneMANIA_ex':
        string_nets = [f'{n}_gm' for n in ['neighborhood', 'fusion',
                                           'cooccurence',
                                           'coexpression',  'experimental',
                                           'database']]
        for network in lst:
            net_path = f"data/raw/{binomial}/{network}"
            net_name = net_path.split('/')[-1].replace('.txt', '')+'_gm'
            string_nets.append(net_name)
        string_nets = sorted(string_nets)
    return string_nets


def textread(filename, astype=str):
    output = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            i = astype(line.strip())
            output.append(i)
    return output


def load_matrix(file_name, shape):
    output = np.zeros(shape)
    with open(file_name, 'r') as f:
        for line in f.readlines():
            g = int(line.strip().split()[0])
            t = int(line.strip().split()[1])
            output[t, g] = 1
    return output



def load_go(org, genes):
    # genes: cell array of query gene names
    go_path = f'data/annotations/{org}'
    go_genes = []
    go_genes = textread(f'{go_path}/{org}_go_genes.txt')

    filt = np.isin(genes, go_genes)
    print(f'genes: {len(genes)}')
    print(f'go_genes: {len(go_genes)}')
    print(f'intersection: {filt.sum()}')
    go_terms = textread(f'{go_path}/{org}_go_terms.txt')

    go_anno = load_matrix(
        f'{go_path}/{org}_go_adjacency.txt', (len(go_terms), len(go_genes)))

    anno = np.zeros((len(go_terms), len(genes)))
    genemap = {ge: i for i, ge in enumerate(go_genes)}
    s2goind = [genemap[ge] for ge in np.array(genes)[filt]]
    anno[:, filt] = go_anno[:, s2goind]
    return anno


def load_network(network_file=None, ngene=None, sym=True):
    """
    load network matrix from text file
    """

    # sparse_network_file = network_file.replace('.txt', '_A.npz')
    # if not os.path.exists(sparse_network_file):
    A = np.zeros((ngene, ngene), dtype='float32')
    with open(network_file, 'r') as f:
        for line in f.readlines():
            x, y, n = line.split()
            x, y, n = int(x), int(y), np.abs(float(n))
            A[x, y] = n

    if sym:
        if (A == np.transpose(A)).sum() != ngene**2:
            A = A + np.transpose(A)

    # np.fill_diagonal(A, 1)
    # A = A - diag(diag(A)
    # if only 0 in one line, assign 1 to diag

    A = A + np.diag(sum(A) == 0)
    adjma = A
    adjma = adjma / (np.expand_dims(np.sum(adjma, axis=1), axis=1) + 0)
    # adjma_sparse = csr_matrix(adjma)
    # save_npz(sparse_network_file, adjma_sparse)
    # del(adjma_sparse)
    del(A)
    # else:
    #     adjma_sparse = load_npz(sparse_network_file)
    #     adjma = adjma_sparse.todense()
    #     del(adjma_sparse)

    return adjma


def min_max(x):
    if len(x.shape) == 1:
        maxval = np.max(x)
        minval = np.min(x)
    else:
        maxval = np.expand_dims(np.max(x, axis=1), axis=1)
        minval = np.expand_dims(np.min(x, axis=1), axis=1)
    x = (x - minval) * (1 / (maxval - minval))
    return x



def out_moment_emb(data, idx):
    network_files, average_type, ngene = data
    network_file = network_files[idx]

    sparse_network_file = network_file.replace('txt', 'npz')
    dense_network_file = network_file.replace('txt', 'npy')
    # print('init', time.time()-s)
    if os.path.exists(sparse_network_file) or \
            os.path.exists(dense_network_file):
        # print(time.time()-s)
        if os.path.exists(dense_network_file):
            # print('load', dense_network_file)
            Q = np.load(dense_network_file)
            # print('loaded', dense_network_file)
        else:
            # print('load', sparse_network_file)
            Q_sparse = load_npz(sparse_network_file)
            # print(time.time()-s)
            Q = Q_sparse.todense()
            del(Q_sparse)
    else:
        print(f'{sparse_network_file} or {dense_network_file} not exist')

    output = []
    for R in [Q]:
        # for R in [G, N]:
        R = np.array(R)
        for od in [1, 2, 3, 4]:
            # s0 = moment(R, moment=2, axis=0)**(od/2)
            s1 = moment(R, moment=2, axis=1)**(od/2)
            # ma0 = moment(R, moment=od, axis=0)
            ma1 = moment(R, moment=od, axis=1)
            # output.extend([ma1])
            output.extend([ma1/s1, ma1])
            del(s1, ma1)
            # del(s0, ma0)
            # output.extend([kurtosis(R, axis=0), kurtosis(R, axis=1)])
    del(R, Q)
    # del(A, Q, R)
    return output
    # return Q
