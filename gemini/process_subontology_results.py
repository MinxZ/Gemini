"""
Given model results, stratify by GO sub-ontology.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import KFold
from tqdm import tqdm
from importlib import reload
import json
import argparse

sys.path.append(os.path.join(sys.path[0], '../'))
from plot import plot_settings, plot_utils
from gemini.func import textread
from config import GEMINI_DIR


# load map_go_id_ont
go_naming = {'molecular_function': 'MF',
             'biological_process': 'BP',
             'cellular_component': 'CC'}

with open(GEMINI_DIR + 'data/map_go_id_ont.txt', 'r') as f:
    go_ids, go_ont = [], []
    for line in f.readlines():
        go_ids.append(line.strip().split()[0])
        go_ont.append(go_naming[line.strip().split()[1]])

# map each GO id to its sub-ontology
subontology2terms = {'MF': [], 'BP': [], 'CC': []}
for term, subont in zip(go_ids, go_ont):
    subontology2terms[subont].append(term)


def compute_metrics(y, yhat):
    precision, recall, thresholds = precision_recall_curve(
        y.flatten(), yhat.flatten())
    f1_scores = 2*recall*precision/(recall+precision+1e-6)
    max_f1 = np.max(f1_scores)

    micro_auprc = auc(recall, precision)

    class_auprcs = []
    for i in range(yhat.shape[1]):
        if y[:, i].sum() != 0:
            precision, recall, thresholds = precision_recall_curve(
                y[:, i], yhat[:, i])
            class_auprcs.append(auc(recall, precision))
    macro_auprc = np.mean(class_auprcs)
    assert max_f1 == max_f1
    return max_f1, micro_auprc, macro_auprc


def get_performance(method, org, net='BioGrid'):
    resroot = GEMINI_DIR + 'results/{}/{}/{}_'.format(net, method.upper(), org.lower())
    y = np.load(resroot + 'labels.npy')  # organized as N_genes x N_functions
    y_hat = np.load(resroot + 'pred.npy')
    test_genes = np.load(resroot + 'testids.npy')
    
    kf = KFold(n_splits=5, random_state=1, shuffle=True) # get our k test folds
    go_terms_path = GEMINI_DIR + f'data/annotations/{org}/{org}_go_terms.txt'
    go_terms = textread(go_terms_path)
    
    train_test_splits = kf.split(range(len(y)))
    train_idxs, test_idxs = [], [] # can only iterate through splits once
    for tr, te in train_test_splits:
        train_idxs.append(tr)
        test_idxs.append(te)
        
    perf_by_subont = {}
    for subont in subontology2terms:
        print('\t...', subont)
        filt = np.isin(go_terms, subontology2terms[subont])  # restrict to the go terms in subontology
        subontology_perf = {'f1': [], 'micro': [], 'macro': []}
        for train, test in zip(train_idxs, test_idxs):
            test_y = y[test][:, filt]
            test_yhat = y_hat[test][:, filt]
            f1, micro, macro = compute_metrics(test_y, test_yhat)
            subontology_perf['f1'].append(f1)
            subontology_perf['micro'].append(micro)
            subontology_perf['macro'].append(macro)
        perf_by_subont[subont] = subontology_perf
    return perf_by_subont


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, help='Input network collection: BioGrid, String, or Combo')
    parser.add_argument('--method', type=str, help='Method to evaluate: Gemini, Mashup, Bionic, Average_Mashup, PCA, or SVD')
    args = parser.parse_args()
    
    if args.method.upper() != "BIONIC" or args.network == 'String': # run everything
        orgs = ['mouse', 'human', 'yeast']
    else:
        orgs = ['yeast']
        
    performance = {}
    for org in orgs:
        print('...evaluating', org)
        performance[org] = get_performance(args.method, org, args.network)
    
    with open(GEMINI_DIR + 'results/{}/{}/results_by_subontology.txt'.format(args.network, args.method.upper()), 'w') as f:
        json.dump(performance, f)
