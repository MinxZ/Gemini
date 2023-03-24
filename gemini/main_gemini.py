"""
Mingxin Zhang
main.py for dca
"""

import argparse
import os
import random
import time
import json

import numpy as np
import torch
from func import out_string_nets, textread
from mashup import load_multi, mashup, mashup_multi
import sys

sys.path.append(os.path.join(sys.path[0], '../'))
from config import GEMINI_DIR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='DCA')
    parser.add_argument('--org', type=str, default='yeast')
    parser.add_argument('--net', type=str, default='GeneMANIA_ex')
    parser.add_argument('--ndim', type=int, default=800)
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
    parser.add_argument('--npz_exist', type=int, default=1)
    parser.add_argument('--ori_seed', type=int, default=0)
    parser.add_argument('--rwr', type=str, default='rwr')

    return parser.parse_args()


args = get_args()


def main():
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    org = args.org

    net = args.net

    method = args.method
    torch_thread = args.torch_thread
    num_thread = args.num_thread

    mixup = True

    ndim = args.ndim

    string_nets = out_string_nets(net, org)

    network_files = []
    for i in range(len(string_nets)):
        network_files.append(GEMINI_DIR + f'data/networks/{org}/{org}_string_{string_nets[i]}_adjacency.txt')

    # Load gene list
    gene_file = GEMINI_DIR + f'data/networks/{org}/{org}_{net}_genes.txt'
    genes = textread(gene_file)
    ngene = len(genes)

    # Mashup integration
    start_time = time.time()
    print(f'{method}_{org}_{net}_{ndim}')
    print('[Mashup]')

    if not os.path.exists(GEMINI_DIR + 'data/embed'):
        os.mkdir(GEMINI_DIR + 'data/embed')

    embd_name = GEMINI_DIR + f'data/embed/{method}_{org}_{net}_{ndim}'

    node_weights = None
    if args.weight > 0 and args.separate != 0:
        weights = np.zeros(len(network_files))
        for embed_type in [args.embed_type]:
            if args.axis > 1:
                iters = [0, 1]
            else:
                iters = [args.axis]
            for axis in iters:
                separate = np.load(
                    GEMINI_DIR + f'data/separate/{net}_{org}_type0_' +
                    f'{embed_type}{axis}_{args.cluster_method}_' +
                    f'{args.level}.npy')
                if args.weight == 2:
                    clus_count = np.ones(len(set(separate)))
                elif args.weight == 1:
                    clus_count = np.zeros(len(set(separate)))
                separate = separate[:len(network_files)]
                for i in separate:
                    clus_count[i] += 1
                if args.weight == 2:
                    clus_weight = 1/clus_count + \
                        args.ori_weight/len(network_files)
                elif args.weight == 1:
                    clus_weight = 1/clus_count
                weights += np.array([clus_weight[i] for i in separate])
                # weights = weights/weights.sum()*len(weights)

        embd_name += f'_{args.embed_type}{args.axis}_' + \
            f'separate{args.separate}_{args.cluster_method}' + \
            f'_weight{args.weight}_{args.ori_weight}'
        args.separate = None
        embd_name += f'_{args.level}'

        if args.mixup > 0:
            network_pairs_mixup_ = []
            from numpy.random import choice
            random.seed(1)
            print(args.ndim)
            # np.random.seed(1)
            p = weights
            p = p/p.sum()
            list_of_candidates = np.arange(len(network_files))
            for idd in range(args.mixup):
                network_pairs_mixup = []
                args.ori_seed = int(np.floor(args.ori_seed*10000)/10000)
                np.random.seed(idd+args.ori_seed)
                for _ in range(round(len(network_files)*args.mixup2)):
                    # for ixd in range(args.mixup):
                    draw = choice(list_of_candidates, 2,
                                  p=p)
                    d0, d1 = draw[0], draw[1]
                    # if separate[d0] != separate[d1]:
                    n0 = network_files[d0]
                    n1 = network_files[d1]
                    network_pairs_mixup.append([n0, 1, n1, 1])
                network_pairs_mixup_.append(network_pairs_mixup)
            mixup = 'mixup'
            embd_name += f'_mixup{args.mixup}_{args.mixup2}'
            embd_name += f'_gamma{args.gamma}'
            network_files_all = network_pairs_mixup_
    elif args.separate == '0':
        args.separate = None

    print('mixup', args.mixup)
    rwr = args.rwr
    if args.mixup < 0:
        mixup = 'average'
        weights = None

        if args.mixup == -1:
            embd_name += '_average'
        else:
            rwr = args.rwr
            embd_name += '_'+rwr
    elif args.mixup == 0:
        mixup = None
        weights = None
    else:
        rwr = 'rwr'

    print(embd_name)
    npz_exist = False if mixup == 'average' else True
    xs = []
    if not os.path.exists(embd_name + '.npy'):
        print(embd_name + '.npy')
        if num_thread == 1:
            # num_thread == 1:
            if args.separate is None:
                x = mashup(network_files, ngene, ndim,
                           mixup, torch_thread, weights)
            else:
                xs = []
                ndim = ndim//len(set(args.separate))
                for sep in set(args.separate):
                    filt = args.separate == sep
                    curr_net = np.array(network_files)[filt]
                    x = mashup(curr_net, ngene, ndim,
                               mixup, torch_thread, weights)
                    xs.append(x)
        else:
            # multi thread
            # num_thread == 1:
            if args.separate is None:
                # weighted on network or no weight
                if npz_exist:
                    # using npz and torch to load faster
                    if mixup != 'mixup':
                        x = load_multi(network_files, ngene, ndim,
                                       mixup, num_thread, torch_thread,
                                       weights,
                                       node_weights=node_weights,
                                       gamma=args.gamma)
                    else:
                        print('Using multiply time mixup to form embeding')
                        xs = []
                        for network_files in network_files_all:
                            xs.append(load_multi(network_files, ngene, ndim,
                                                 mixup, num_thread,
                                                 torch_thread,
                                                 weights,
                                                 node_weights=node_weights,
                                                 gamma=args.gamma))

                else:
                    x = mashup_multi(network_files, ngene, ndim,
                                     mixup, num_thread, torch_thread,
                                     weights,
                                     node_weights=node_weights,
                                     rwr=rwr)

            else:
                # weighted on nodes
                xs = []
                ndim = ndim//len(set(args.separate))
                for sep in set(args.separate):
                    filt = args.separate == sep
                    curr_net = np.array(network_files)[filt]
                    x = mashup_multi(curr_net, ngene, ndim,
                                     mixup, num_thread, torch_thread,
                                     weights, node_weights=node_weights)
                    xs.append(x)

        if len(xs) > 0:
            x = np.concatenate(xs, axis=0)
        np.save(embd_name, x)
        end_time = time.time()
        print(f'Time: {end_time-start_time}')
        print(x)


if __name__ == '__main__':
    main()
