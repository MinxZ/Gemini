"""
Mingxin Zhang
main.py for dca
"""

import argparse
import os
import random
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

from func import out_moment_emb, out_string_nets, textread
from mashup import mashup_multi


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='DCA')
    parser.add_argument('--org', type=str, default='mouse')
    parser.add_argument('--net', type=str, default='GeneMANIA')
    parser.add_argument('--ndim', type=int, default=500)
    parser.add_argument('--num_thread', type=int, default=5)
    parser.add_argument('--torch_thread', type=int, default=4)
    parser.add_argument('--run_mashup', type=int, default=0)
    parser.add_argument('--separate', type=str, default=35)
    parser.add_argument('--cluster_method', type=str, default='app')
    parser.add_argument('--level', type=str, default='network')
    parser.add_argument('--embed_type', type=str, default='Qsm4')
    parser.add_argument('--axis', type=int, default=1)
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

    # Load gene list
    gene_file = f'data/networks/{org}/{org}_{net}_genes.txt'
    genes = textread(gene_file)
    ngene = len(genes)

    network_files = []
    for i in range(len(string_nets)):
        network_files.append(
            f'data/networks/{org}/{org}_string_{string_nets[i]}_adjacency.txt')

    num_net = len(network_files)
    print(num_net)
    average_type = 0
    # 0 averaged on log rwr
    # 1 averaged on log rwr
    # compute cluster
    embeds = []
    embed_name = f'data/embed/{net}_{org}_type{average_type}_' + \
        f'{args.embed_type}{args.axis}_{args.level}.npy'

    if os.path.exists(
            embed_name) or 'all' in embed_name:
        pass
    else:
        print(embed_name)
        print('calculate embedding for each network')
        run_mashup = args.run_mashup
        if run_mashup == 1:
            # Mashup integration
            print(f'{method}_{org}_{net}_{ndim}')
            print('[Mashup]')
            _ = mashup_multi(network_files, ngene, ndim,
                             mixup, num_thread, torch_thread)

        if args.level == 'network':
            data = network_files, average_type, ngene
            f = partial(out_moment_emb, data)
            max_len = num_net
        elif args.level == 'node':
            data = network_files, average_type, ngene
            # f = partial(out_var_emb_node, data)
            max_len = ngene

        max_idx = max_len//num_thread
        max_idx = max_idx+1 if max_len % num_thread > 0 else max_idx
        for i in tqdm(range(max_idx)):
            start_idx = num_thread*(i)
            end_idx = num_thread*(i+1)
            end_idx = end_idx if end_idx <= max_len else max_len
            idxs = np.arange(num_net)[start_idx:end_idx]
            use_pool = True
            if not use_pool:
                embed_part = []
                for idx in idxs:
                    embed_part.append(f(idx))
            else:
                with Pool(processes=num_thread) as pl:
                    embed_part = pl.map(f, idxs)
            embeds.extend(embed_part)

        i_ = -1
        for od in [1, 2, 3, 4]:
            for em in ['Q']:
                for embed_type in [f'{em}sm{od}', f'{em}m{od}']:
                    for axis in [1]:
                        i_ += 1
                        c = np.array([embeds[i][i_]
                                      for i in range(len(embeds))])
                        np.save(
                            f'data/embed/{net}_{org}_type{average_type}_' +
                            f'{embed_type}{axis}_{args.level}', c)
    c = np.load(
        f'data/embed/{net}_{org}_type{average_type}_' +
        f'{args.embed_type}{args.axis}_{args.level}.npy')
    c = c[:len(network_files)]
    print(c.shape)
    n_components = int(args.separate)

    cluster = args.cluster_method
    if cluster == 'gm':
        from sklearn.mixture import GaussianMixture
        print('Run GaussianMixture')
        clustering = GaussianMixture(
            n_components=n_components, random_state=0).fit(c)
        separate = clustering.predict(c)

    elif cluster == 'sc':
        from sklearn.cluster import SpectralClustering
        print('RunSpectralClustering')
        clustering = SpectralClustering(n_clusters=n_components,
                                        assign_labels='discretize',
                                        random_state=0).fit(c)
        separate = clustering.labels_
    elif cluster == 'km':
        from sklearn.cluster import MiniBatchKMeans
        clustering = MiniBatchKMeans(n_clusters=n_components,
                                     random_state=0,
                                     batch_size=10,
                                     max_iter=100).fit(c)
        separate = clustering.labels_
    elif cluster == 'op':
        from sklearn.cluster import OPTICS
        clustering = OPTICS(min_samples=n_components).fit(c)
        separate = clustering.labels_
    elif cluster == 'ms':
        from sklearn.cluster import MeanShift
        clustering = MeanShift().fit(c)
        separate = clustering.labels_
    elif cluster == 'ap':
        from sklearn.cluster import AffinityPropagation
        print('run AffinityPropagation')
        clustering = AffinityPropagation(
            damping=n_components/40, random_state=0).fit(c)
        separate = clustering.labels_
    elif cluster == 'app':
        from sklearn.metrics.pairwise import euclidean_distances
        net_seq = []
        for network_file in tqdm(np.array(network_files)):
            seq1, seq2 = [], []
            with open(network_file, 'r') as f:
                for line in f.readlines():
                    x, y, n = line.split()
                    x, y, n = int(x), int(y), np.abs(float(n))
                    seq1.append(x)
                    seq2.append(y)
            net_seq.append(set(seq1+seq1))
        N_graphs = len(net_seq)
        kurt_dist = np.zeros((N_graphs, N_graphs))
        for g1 in tqdm(range(N_graphs)):
            for g2 in range(g1+1, N_graphs):
                common = list(net_seq[g1].intersection(net_seq[g2]))
                if len(common) > 0:
                    sim = euclidean_distances(c[g1][common].reshape(1, -1),
                                              c[g2][common].reshape(1, -1))
                    kurt_dist[g1, g2] = sim
                    kurt_dist[g2, g1] = sim
        means = kurt_dist[kurt_dist > 0].mean()
        kurt_dist[kurt_dist == 0] = means
        np.fill_diagonal(kurt_dist, 0)

        from sklearn.cluster import AffinityPropagation
        print('run AffinityPropagation')
        clustering = AffinityPropagation(
            affinity='precomputed',
            damping=n_components/40, random_state=0).fit(kurt_dist)
        separate = clustering.labels_
    elif cluster == 'acp':
        from sklearn.metrics.pairwise import euclidean_distances
        net_seq = []
        for network_file in tqdm(np.array(network_files)):
            seq1, seq2 = [], []
            with open(network_file, 'r') as f:
                for line in f.readlines():
                    x, y, n = line.split()
                    x, y, n = int(x), int(y), np.abs(float(n))
                    seq1.append(x)
                    seq2.append(y)
            net_seq.append(set(seq1+seq1))
        N_graphs = len(net_seq)
        kurt_dist = np.zeros((N_graphs, N_graphs))
        for g1 in tqdm(range(N_graphs)):
            for g2 in range(g1+1, N_graphs):
                common = list(net_seq[g1].intersection(net_seq[g2]))
                if len(common) > 0:
                    sim = euclidean_distances(c[g1][common].reshape(1, -1),
                                              c[g2][common].reshape(1, -1))
                    kurt_dist[g1, g2] = sim
                    kurt_dist[g2, g1] = sim
        # kurt_dist = np.array(kurt_dist_seq)
        means = kurt_dist[kurt_dist > 0].mean()
        kurt_dist[kurt_dist == 0] = means
        np.fill_diagonal(kurt_dist, 0)

        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            affinity='precomputed',
            n_clusters=n_components,
            linkage='average').fit(kurt_dist)
        separate = clustering.labels_

    print(separate)
    print(set(separate))
    print(len(set(separate)))
    num2i = {num: i for i, num in enumerate(list(set(separate)))}
    separate = [num2i[num] for num in separate]

    if not os.path.exists('data/separate'):
        os.mkdir('data/separate')
    np.save(
        f'data/separate/{net}_{org}_type{average_type}_' +
        f'{args.embed_type}{args.axis}_{cluster}_{args.level}',
        separate)


if __name__ == '__main__':
    main()
