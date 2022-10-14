"""
Mingxin Zhang
"""

import argparse
import random

import numpy as np

from load_anno_vali import load_anno_and_cross_validation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='NN')
    parser.add_argument('--org', type=str, default='yeast')
    parser.add_argument('--ndim', type=int, default=200)
    parser.add_argument('--base', type=int, default=200)
    parser.add_argument('--ori_ndim', type=int, default=200)
    parser.add_argument('--net', type=str, default='GeneMANIA_ex')
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--experiment_name', type=str,
                        default='gemini_yeast_GeneMANIA_ex_200_NN_Qsm41_' +
                        'separate35_ap' +
                        '_weight1_0.5_network_mixup1_1.0_gamma0.5_seed1')
    parser.add_argument('--embed_name', type=str,
                        default='gemini_yeast_GeneMANIA_ex_200_Qsm41_' +
                        'separate35_ap_weight1_0.5_network_mixup1_1.0' +
                        '_gamma0.5')
    parser.add_argument('--best_epoch', type=int,
                        default=None)
    parser.add_argument('--mixup', type=int,
                        default=0)
    parser.add_argument('--seed', type=int,
                        default=-1)
    return parser.parse_args()


args = get_args()


def main():
    random.seed(1)
    model_type = args.model_type
    org = args.org
    net = args.net
    embed_name = args.embed_name
    experiment_name = f'{args.experiment_name}_{args.ndim}'
    ratio = args.ratio
    best_epoch = args.best_epoch
    base = args.base
    seed = args.seed
    if args.mixup == 0:
        x = np.load(f'data/embed/{embed_name}.npy')[:args.ndim, :]
    elif args.mixup in [1, 5]:
        # full
        x = np.load(f'data/embed/{embed_name}.npy')
        if not args.seed == -1:
            x = x[(seed-1)*args.ndim:(seed)*args.ndim, :]
    elif args.mixup == 6:
        # average
        x = np.load(f'data/embed/{embed_name}.npy')
        xs = np.zeros((args.ndim, x.shape[1]))
        for i in range(5):
            xs += x[i*base:i*base+args.ndim, :]
        x = xs
        x /= 5
    elif args.mixup == 7:
        # concatenate
        x = np.load(f'data/embed/{embed_name}.npy')
        xs = []
        for i in range(5):
            xs.append(x[i*base:i*base+args.ndim, :])
        x = np.concatenate(xs, axis=0)
    print(x.shape)
    load_anno_and_cross_validation(model_type, org, net, experiment_name,
                                   x, ratio, best_epoch)


if __name__ == '__main__':
    main()
