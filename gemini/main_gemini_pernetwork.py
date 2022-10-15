"""
Mingxin Zhang
main.py for dca
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch

from func import textread
from mashup import mashup_vali


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='DCA')
    parser.add_argument('--org', type=str, default='yeast')
    parser.add_argument('--net', type=str, default='GeneMANIA')
    parser.add_argument('--ndim', type=int, default=200)
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--torch_thread', type=int, default=5)
    parser.add_argument('--weight', type=int, default=0)
    parser.add_argument('--separate', type=str, default='0')
    parser.add_argument('--ori_weight', type=float, default=0.5)
    parser.add_argument('--cluster_method', type=str, default='ap')
    parser.add_argument('--level', type=str, default='network')
    parser.add_argument('--embed_type', type=str, default='Qsm4')
    parser.add_argument('--axis', type=int, default=1)
    parser.add_argument('--mixup', type=int, default=0)
    parser.add_argument('--mixup2', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--best_epoch', type=int, default=5)
    return parser.parse_args()


args = get_args()


def main():
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    org = args.org
    net = args.net
    ndim = args.ndim

    if net == 'GeneMANIA' or 'ex' in net:
        org2binomial = {'yeast': 'Saccharomyces_cerevisiae',
                        'human': 'Homo_sapiens',
                        'mouse': 'Mus_musculus'}

        binomial = org2binomial[org]
        string_nets = []
        lst = pd.read_csv(f"data/raw/{binomial}/networks.txt",
                          sep='\t').File_Name
        string_nets = []

    if net == 'net12':
        data = pd.read_csv('data/raw/drug/data1.csv')
        seq3 = list(data['data_source(s)'])
        seq3 = [set(item.split(',')) for item in seq3]
        string_nets = list(set([j for i in seq3 for j in i]))
        string_nets = sorted(string_nets)
        # string_nets = [string_nets[0]] + string_nets[2:]
    elif net == 'GeneMANIA':
        for network in lst:
            net_path = f"data/raw/{binomial}/{network}"
            net_name = net_path.split('/')[-1].replace('.txt', '')
            string_nets.append(net_name)
        string_nets = sorted(string_nets)
    elif net == 'mashup':
        string_nets = ['neighborhood', 'fusion', 'cooccurence',
                       'coexpression',  'experimental', 'database']
        string_nets = sorted(string_nets)

    network_files = []
    for i in range(len(string_nets)):
        network_files.append(
            f'data/networks/{org}/{org}_string_{string_nets[i]}_adjacency.txt')

    # Load gene list
    gene_file = f'data/networks/{org}/{org}_{net}_genes.txt'
    genes = textread(gene_file)
    ngene = len(genes)

    mashup_vali(org, net, network_files, ngene,
                best_epoch=args.best_epoch, torch_thread=12, ndim=ndim)


if __name__ == '__main__':
    main()
