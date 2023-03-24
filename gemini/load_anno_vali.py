import os
import torch
import numpy as np
import json
import sys
import os

sys.path.append(os.path.join(sys.path[0], '../'))
from config import GEMINI_DIR
from gemini.cross_validation_nn import cross_validation_nn
from gemini.func import load_go, load_IntAct, textread


def load_anno(org, net):
    # Load known annotations
    # alpha is a parameter used to calculate f1
    print('[Loading annotations]\n')
    if net != 'bionic':
        gene_file = GEMINI_DIR + f'data/networks/{org}/{org}_{net}_genes.txt'
        genes = textread(gene_file)
        anno = load_go(org, genes)
    else:
        print('LOADING BIONIC ANNOTATIONS')
        with open(GEMINI_DIR + 'data/networks/bionic/bionic_gene_ordering.txt', 'r') as f:
            genes = json.load(f)
            genes = sorted(list(genes.keys()), key=lambda g: genes[g])
        anno = load_IntAct(genes)
    print('Number of functional labels: %d\n' % (anno.shape[1-1]))
    return anno


def load_anno_and_cross_validation(model_type, org, net, experiment_name,
                                   x, ratio, best_epoch, batch_size=128, device=None):
    """
    params:
    model_type: SVM or SVR, NN, recommend NN
    org: yeast, human, mouse, drug
    net: GeneMANIA, string
    experiment_name: to save auprc, and gmax, cmax
    num_thread: 0 means using all thread
    ratio: default 0.2, test data ratio
    """
    if device is None:
#         if torch.backends.mps.is_available():
#             device = torch.device('mps')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif type(device) == str:
        device = torch.device(device)

    if not os.path.exists(GEMINI_DIR + 'data/results/'):
        os.mkdir(GEMINI_DIR + 'data/results/')
    if not os.path.exists(GEMINI_DIR + 'data/results/raw'):
        os.mkdir(GEMINI_DIR + 'data/results/raw')
    print(GEMINI_DIR + f'data/results/{experiment_name}' +
          f'_{best_epoch}_result.txt')
    if os.path.exists(GEMINI_DIR + f'data/results/{experiment_name}' +
                      f'_{best_epoch}_result.txt'):
        pass
    else:
        np.random.seed(1)
        # number of cross-validation trials
        nperm = 5
        # Load gene list
        if net == 'bionic':
            gene_file = GEMINI_DIR + 'data/networks/bionic/bionic_gene_ordering.txt'
            with open(gene_file, 'r') as f:
                genes = json.load(f)
                genes = sorted(list(genes.keys()), key=lambda g: genes[g])
        else:
            gene_file = GEMINI_DIR + f'data/networks/{org}/{org}_{net}_genes.txt'
            genes = textread(gene_file)

        # Load known annotations
        # alpha is a parameter used to calculate f1
        print('[Loading annotations]\n')
        if net != 'bionic':
            anno = load_go(org, genes)
        else:
            anno = load_IntAct(genes)
        alpha = 3
        print('Number of functional labels: %d\n' % (anno.shape[1-1]))

        # Function prediction
        print(f'Model type: {model_type}')
        print('[Function prediction]\n')

        _, _, _, _, acc, f1, aupr, roc, preds, labels, test_ids = \
            cross_validation_nn(
                x, anno, nperm, batch_size=batch_size,
                alpha=alpha, ratio=ratio,
                best_epoch_=best_epoch, return_pred=True,
                device=device)

        np.save(GEMINI_DIR + f'data/results/raw/{experiment_name}_pred',
                np.concatenate(preds, axis=0))
        np.save(GEMINI_DIR + f'data/results/raw/{experiment_name}_labels',
                np.concatenate(labels, axis=0))
        np.save(GEMINI_DIR + f'data/results/raw/{experiment_name}_testids',
                np.concatenate(test_ids, axis=0))

        # Output summary
        output = []
        output.append('[Performance]')
        output.append(f'Epoch {best_epoch}')
        output.append('Accuracy: %f (stdev = %f)' %
                      (np.mean(acc), np.std(acc)))
        output.append('F1: %f (stdev = %f)' % (np.mean(f1), np.std(f1)))
        output.append('AUPRC: %f (stdev = %f)' % (np.mean(aupr), np.std(aupr)))
        output.append('MAPRC: %f (stdev = %f)' % (np.mean(roc), np.std(roc)))

        with open(GEMINI_DIR + f'data/results/{experiment_name}' +
                  f'_{best_epoch}_result.txt', 'w') \
                as f:
            for line in output:
                print(line)
                f.writelines(line+'\n')
        with open(GEMINI_DIR + f'data/results/raw/{experiment_name}' +
                  f'_{best_epoch}_acc.txt', 'w') \
                as f:
            for line in acc:
                line = str(line[0])
                print(line)
                f.writelines(line+'\n')
        with open(GEMINI_DIR + f'data/results/raw/{experiment_name}' +
                  f'_{best_epoch}_auprcs.txt', 'w') \
                as f:
            for line in aupr:
                line = str(line[0])
                print(line)
                f.writelines(line+'\n')
        with open(GEMINI_DIR + f'data/results/raw/{experiment_name}' +
                  f'_{best_epoch}_f1.txt', 'w') \
                as f:
            for line in f1:
                line = str(line[0])
                print(line)
                f.writelines(line+'\n')
        with open(GEMINI_DIR + f'data/results/raw/{experiment_name}' +
                  f'_{best_epoch}_roc.txt', 'w') \
                as f:
            for line in roc:
                line = str(line[0])
                print(line)
                f.writelines(line+'\n')
