"""
Mingxin Zhang
"""

import argparse
import random

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(sys.path[0], '../'))
from gemini.load_anno_vali import load_anno_and_cross_validation
from config import GEMINI_DIR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='NN')
    parser.add_argument('--org', type=str)
    parser.add_argument('--net', type=str)
    parser.add_argument('--config-name', type=str)
    parser.add_argument('--ratio', type=float, default=0.2)
    return parser.parse_args()


args = get_args()


def main():
    random.seed(1)
    model_type = args.model_type
    org = args.org
    net = args.net
    assert net in {'String', 'BioGrid', 'Combo'}
    net = 'mashup_ex' if net == 'String' else 'GeneMANIA_ex' if net == 'BioGrid' else 'mashup_GeneMANIA_ex'
    config = args.config_name

    with open(GEMINI_DIR + 'data/networks/{}/{}_{}_genes.txt'.format(org, org, net), 'r') as f:
        num_genes = len(f.readlines())
    features = pd.read_csv('config_files/{}_{}_{}_features.tsv'.format(args.net, org, config), delimiter='\t')
    # convert this dataframe feature to np matrix
    feature_mtx = features.to_numpy()[:, 1:]
    x = np.zeros((num_genes, len(features.columns[1:])))
    print(x.shape)
    for i in range(len(features)):
        gene_idx = features['Unnamed: 0'].loc[i]
        x[gene_idx] = feature_mtx[i]

    x = x.T
    print('...feature shape:', x.shape)
    load_anno_and_cross_validation(model_type, org, net, '{}_{}_{}'.format(args.net, org, config),
                                   x, args.ratio, None)  # don't give best epochs -> we shouldn't need this!


if __name__ == '__main__':
    main()
