# network_files (cell array): paths to adjacency list files
# ngene (int): number of genes in input networks
# ndim (int): number of output dimensions
# mixup (bool): whether to use SVD approximation for large-scale networks
##

import os
import sys
import random
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch

sys.path.append(os.path.join(sys.path[0], '../'))
from gemini.cross_validation_nn import validation_nn_output
from gemini.func import load_network
from joblib import Parallel, delayed
from gemini.load_anno_vali import load_anno
from gemini.rwr_func import rwr, rwr_torch
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.linalg import eigsh, svds
from sklearn.decomposition import PCA
from tqdm import tqdm

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


def network_svd(ndim, torch_thread, RR_sum, verbose=1):
    s = time.time()
    torch.set_num_threads(torch_thread)
    if verbose == 1:
        print('All networks loaded. Learning vectors via SVD...\n')
    torch.manual_seed(1)
    try:
        d, V = torch.linalg.eigh(torch.Tensor(RR_sum))
        del(RR_sum)
        V = V.numpy()[:, ::-1][:, :ndim]
        d = d.numpy()[::-1][:ndim]
        x = np.diag(np.sqrt(np.sqrt(d))).dot(V.T)
        del(d, V)
    except:
        d, V = eigsh(RR_sum, k=ndim)
        del(RR_sum)
        V = V[:, ::-1]
        d = d[::-1]
        x = np.diag(np.sqrt(np.sqrt(d))).dot(np.transpose(V))

    if verbose == 1:
        print(time.time()-s)
        print('Mashup features obtained.\n')
    return x


def load_and_rwr(ngene, torch_thread, network_file, alpha=0.5):
    s = time.time()
    use_torch = True
    torch.set_num_threads(torch_thread)
    torch.manual_seed(1)
    np.random.seed(1)
    # random.seed(1)
    sparse_network_file = network_file.replace('txt', 'npz')
    dense_network_file = network_file.replace('txt', 'npy')
    # print('init', time.time()-s)
    if os.path.exists(sparse_network_file) or \
            os.path.exists(dense_network_file):
        # print(time.time()-s)
        if os.path.exists(dense_network_file):
            # print('load', dense_network_file)
            # Q = np.load(dense_network_file, mmap_mode='r')
            Q = np.load(dense_network_file)
            # print('loaded', dense_network_file)
        else:
            # print('load', sparse_network_file)
            Q_sparse = load_npz(sparse_network_file)
            if Q_sparse.dtype == 'float64':
                Q_sparse = Q_sparse.astype('float32')
                save_npz(sparse_network_file, Q_sparse)
            # print(time.time()-s)
            Q = Q_sparse.todense()
        # print('load Q', time.time()-s)
        # del(Q_sparse)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('devise', time.time()-s)
        # Q = torch.from_numpy(Q).to(device)
        # print('to gpu', time.time()-s)

    else:
        print(sparse_network_file)
        A = load_network(network_file, ngene)
        if A.dtype == 'float64':
            A = A.astype('float32')
        print('load A', time.time()-s)
        if use_torch:
            Q = rwr_torch(A, alpha)
        else:
            Q = rwr(A, alpha)
        print('rwr Q', time.time()-s)

        if Q.dtype == 'float64':
            Q = Q.astype('float32')

        Q_sparse = csr_matrix(Q)
        save_npz(sparse_network_file, Q_sparse)
        del(Q_sparse)
        del(A)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('devise', time.time()-s)
        # Q = torch.from_numpy(Q).to(device)
        # print('to gpu', time.time()-s)

    # print(2)
    Q = np.array(Q)

    # R = Q
    # R = np.log(Q + 1 / ngene)
    # R = torch.log(Q + 1 / ngene)

    # print('log R', time.time()-s)
    # R = R.dot(R.T)
    # R = torch.mm(R.T, R)
    # print('dot RR', time.time()-s)
    # del(Q)
    # R = np.array(R)

    # R = R.cpu().numpy()
    # print('to cpu numpy', time.time()-s)
    # R = R.astype(np.float32)

    return Q


def load_and_mixup_rwr(ngene, torch_thread, gamma, network_file):
    # s = time.time()
    # torch.set_num_threads(torch_thread)
    # torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    n1, w1, n2, w2 = network_file

    # A1 = load_network(n1, ngene)
    # A2 = load_network(n2, ngene)
    # A = (gamma*A1 + (1-gamma)*A2)
    # Q = rwr_torch(A, 0.5)
    # del(A, A1, A2)

    # s1 = n1.replace('txt', 'npz')
    # s2 = n2.replace('txt', 'npz')
    # Q1 = load_npz(s1)
    # Q2 = load_npz(s2)
    Q1 = load_and_rwr(ngene, torch_thread, n1)
    Q2 = load_and_rwr(ngene, torch_thread, n2)
    Q = (gamma*Q1 + (1-gamma)*Q2)
    if type(Q) != np.ndarray:
        Q = Q.todense()
    del(Q1, Q2)
    Q = np.array(Q)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # print('devise', time.time()-s)
    # Q = torch.from_numpy(Q).to(device)

    # # R = np.log(Q + 1 / ngene)
    # R = torch.log(Q + 1 / ngene)

    # # print('log R', time.time()-s)
    # # R = R.dot(R.T)
    # R = torch.mm(R.T, R)
    # # print('dot RR', time.time()-s)
    # del(Q)
    # # R = np.array(R)

    # # R = R.cpu().numpy()
    # # print('to cpu numpy', time.time()-s)
    # # R = R.astype(np.float32)

    return Q


def rwr_load(network_file):
    sparse_network_file = network_file.replace('txt', 'npz')
    # print(time.time()-s)
    Q_sparse = load_npz(sparse_network_file)
    # print(time.time()-s)
    Q = Q_sparse.todense()
    # print('load Q', time.time()-s)
    return Q


def load_adj(ngene, network_file):
    A = load_network(network_file, ngene)
    A = np.array(A)
    return A


def load_and_rwr_weight(ngene, torch_thread, node_weights, network_file):
    use_torch = True
    torch.set_num_threads(torch_thread)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    sparse_network_file = network_file.replace('txt', 'npz')
    if not os.path.exists(sparse_network_file):
        A = load_network(network_file, ngene)
        if use_torch:
            Q = rwr_torch(A, 0.5)
        else:
            Q = rwr(A, 0.5)
        Q_sparse = csr_matrix(Q)
        save_npz(sparse_network_file, Q_sparse)
        del(A)
    else:
        Q_sparse = load_npz(sparse_network_file)
        Q = Q_sparse.todense()
    del(Q_sparse)

    R = np.log(Q + 1 / ngene)
    R *= node_weights
    # R = R.dot(R.T)
    R = R.T.dot(R)
    del(Q)
    R = np.array(R)
    return R


def mashup(network_files=None, ngene=None, ndim=None, mixup=None,
           torch_thread=12, weights=None, separate=None, device=None):
    s = time.time()
    torch.manual_seed(1)
    torch.set_num_threads(torch_thread)
    random.seed(1)
    np.random.seed(1)
    s = time.time()
    # RR_sum = np.zeros((ngene, ngene))
    if device is None:
#         if torch.backends.mps.is_available():
#             device = torch.device('mps')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif type(device) == str:
        device = torch.device(device)

    if device == torch.device('cuda'):
        RR_sum = torch.cuda.FloatTensor(ngene, ngene).fill_(0)
    elif device == torch.device('cpu'):
        RR_sum = torch.FloatTensor(ngene, ngene).fill_(0)
#     elif device == torch.device('mps'):
#         RR_sum = torch.FloatTensor(ngene, ngene).fill_(0).to(device)
    i = 0
    # print('devise', time.time()-s)
    # Q = torch.from_numpy(Q).to(device)
    weights = np.ones(len(network_files)) if weights is None else weights

    if separate is None:
        for network_file in tqdm(network_files):
            R = load_and_rwr(ngene, torch_thread, network_file)
            if weights is None:
                RR_sum += R
            else:
                RR_sum = RR_sum + R * weights[i]
            i += 1
        del(R)
        print(time.time()-s)
        print()
        RR_sum = RR_sum.cpu().numpy()
        x = network_svd(ndim, torch_thread, RR_sum)
    else:
        xs = []
        num_nets = len(network_files)
        for network_file in tqdm(network_files):
            R = load_and_rwr(ngene, torch_thread, network_file)
            xs.append(network_svd(ndim//num_nets, torch_thread, R))
            i += 1
        del(R)
        print(time.time()-s)
        print()
        x = np.concatenate(xs, axis=0)
        del(xs)
    return x


def mashup_vali(org, net, network_files, ngene=None,
                best_epoch=100, torch_thread=12, ndim=None,
                device=None):
    torch.manual_seed(1)
    torch.set_num_threads(torch_thread)
    random.seed(1)
    np.random.seed(1)
    if device is None:
#         if torch.backends.mps.is_available():
#             device = torch.device('mps')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif type(device) == str:
        device = torch.device(device)

    # Load gene list
    anno = load_anno(org, net)
    # Function prediction
    print('[Function prediction]\n')

    preds = []
    for network_file in network_files:
        Q = load_and_rwr(ngene, torch_thread, network_file)
        Q = torch.from_numpy(Q).to(device)
        R = torch.log(Q + 1 / ngene)
        RR_sum = torch.mm(R.T, R)
        RR_sum = RR_sum.cpu().numpy()
        x = network_svd(ndim, torch_thread, RR_sum, verbose=0)
        x = validation_nn_output(
            x, anno,
            best_epoch_=best_epoch)

        np.save(network_file.replace('.txt', '_pred'), x)

    return preds


def mashup_multi(network_files=None, ngene=None, ndim=None,
                 mixup=None, num_thread=5, torch_thread=4,
                 weights=None, separate=None, node_weights=None,
                 rwr='rwr', device=None):
    weights_ = np.ones(len(network_files)) if weights is None else weights
    if device is None:
#         if torch.backends.mps.is_available():
#             device = torch.device('mps')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif type(device) == str:
        device = torch.device(device)

    if device == torch.device('cuda'):
        RR_sum = torch.cuda.FloatTensor(ngene, ngene).fill_(0)
    elif device == torch.device('cpu'):
        RR_sum = torch.FloatTensor(ngene, ngene).fill_(0)
    elif device == torch.device('mps'):
        RR_sum = torch.FloatTensor(ngene, ngene).fill_(0).to(device)

    max_len = len(network_files)
    max_idx = max_len//num_thread
    max_idx = max_idx+1 if max_len % num_thread > 0 else max_idx
    xs = []
    num_nets = len(network_files)
    f2 = partial(network_svd, ndim//num_nets, torch_thread)
    for i in tqdm(range(max_idx)):
        start_idx = num_thread*(i)
        end_idx = num_thread*(i+1)
        end_idx = end_idx if end_idx <= max_len else max_len
        current_networks = network_files[start_idx:end_idx]
        current_weights = weights_[start_idx:end_idx]

        if node_weights is not None:
            # print('node_weight')
            f = partial(load_and_rwr_weight, ngene, torch_thread, node_weights)
        elif mixup == 'mixup':
            # print('mixup')
            f = partial(load_and_mixup_rwr, ngene, torch_thread)
            weights = None
        elif mixup == 'average':
            # print('average')
            f = partial(load_adj, ngene)
            weights = None
        else:
            # print('network_weight')
            f = partial(load_and_rwr, ngene, torch_thread)

        use_pool = True
        if use_pool:
            with Pool(processes=num_thread) as pl:
                RR_sums = pl.map(f, current_networks)
        else:
            RR_sums = Parallel(n_jobs=num_thread, prefer="threads")(
                delayed(f)(current_network) for current_network in
                current_networks)

        if weights is not None:
            RR_sums = [RR_sums[idx]*current_weights[idx]
                       for idx in range(len(RR_sums))]
        # print(len(RR_sums))
        if separate is None:
            if mixup == 'average':
                for idx, Q in enumerate(RR_sums):
                    Q = torch.from_numpy(Q).to(device)
                    RR_sum += Q
            else:
                for idx, Q in enumerate(RR_sums):
                    RR_sums[idx] = torch.from_numpy(Q).to(device)
                RR_sum += torch.stack(RR_sums, dim=0).sum(dim=0)
        else:
            with Pool(processes=num_thread) as pl:
                xs = pl.map(f2, RR_sums)
        del(RR_sums)

    if mixup == 'average':
        A = RR_sum/max_len
        A = A.cpu().numpy()
        if rwr == 'rwr':
            Q = rwr_torch(A, 0.5)
            Q = torch.from_numpy(Q).to(device)
            R = torch.log(Q + 1 / ngene)
        else:
            Q = A.copy()
            Q = torch.from_numpy(Q).to(device)
            if 'log':
                R = torch.log(Q + 1 / ngene)
            else:
                R = Q

            if 'svd' in rwr:
                RR_sum = torch.mm(R.T, R)
            else:
                RR_sum = R

        del(Q)

    if separate is None:
        RR_sum = RR_sum.cpu().numpy()
        if rwr == 'rwr' or 'svd' in rwr:
            x = network_svd(ndim, num_thread*torch_thread, RR_sum)
        elif 'pca' in rwr:
            pca = PCA(n_components=ndim)
            x = pca.fit_transform(RR_sum).T
    else:
        x = np.concatenate(xs, axis=0)
    del(xs)
    return x


def load_multi(network_files=None, ngene=None, ndim=None,
               mixup=None, num_thread=5, torch_thread=4,
               weights=None, separate=None, node_weights=None, gamma=None,
                device=None):
    if device is None:
#         if torch.backends.mps.is_available():
#             device = torch.device('mps')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif type(device) == str:
        device = torch.device(device)

    print('load multi')
    s = time.time()
    torch.set_num_threads(torch_thread)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    weights_ = np.ones(len(network_files)) if weights is None else weights
    if device == torch.device('cuda'):
        RR_sum = torch.cuda.FloatTensor(ngene, ngene).fill_(0)
    elif device == torch.device('cpu'):
        RR_sum = torch.FloatTensor(ngene, ngene).fill_(0)
    elif device == torch.device('mps'):
        RR_sum = torch.FloatTensor(ngene, ngene).fill_(0).to(device)

    max_len = len(network_files)
    max_idx = max_len//num_thread
    max_idx = max_idx+1 if max_len % num_thread > 0 else max_idx
    xs = []
    num_nets = len(network_files)
    f2 = partial(network_svd, ndim//num_nets, torch_thread)
    for i in tqdm(range(max_idx)):
        start_idx = num_thread*(i)
        end_idx = num_thread*(i+1)
        end_idx = end_idx if end_idx <= max_len else max_len
        current_networks = network_files[start_idx:end_idx]

        if node_weights is not None:
            # print('node_weight')
            current_weights = weights_[start_idx:end_idx]
            f = partial(load_and_rwr_weight, ngene, torch_thread, node_weights)
        elif mixup == 'mixup':
            f = partial(load_and_mixup_rwr, ngene, torch_thread, gamma)
            weights = None
        else:
            # print('network_weight')
            current_weights = weights_[start_idx:end_idx]
            f = partial(load_and_rwr, ngene, torch_thread)

        use_pool = 'p1'
        if use_pool == 'p1':
            with Pool(processes=num_thread) as pl:
                Qs = pl.map(f, current_networks)
        elif use_pool == 'p2':
            Qs = Parallel(n_jobs=num_thread, prefer="threads")(
                delayed(f)(current_network) for current_network in
                current_networks)
        elif use_pool == 's':
            Qs = []
            for current_network in current_networks:
                Qs.append(f(current_network))

        # print('devise', time.time()-s)
        RR_sums = []
        # s = time.time()
        for idx, Qcpu in enumerate(Qs):
            if Qcpu.dtype == 'float64':
                Qcpu = Qcpu.astype('float32')
            # del(Q_sparse)
            Q = torch.from_numpy(Qcpu).to(device)
            # print(time.time() - s)
            if mixup == 'average':
                R = Q
            else:
                R = torch.log(Q + 1 / ngene)
                R = torch.mm(R.T, R)
            # print(time.time() - s)
            if weights is not None:
                RR_sum = R * current_weights[idx]
            else:
                RR_sum += R
            # print(time.time() - s)
        del(Qs)
        if separate is None:
            pass
        else:
            with Pool(processes=num_thread) as pl:
                xs = pl.map(f2, RR_sums)
        del(RR_sums)
    if mixup == 'average':
        A = RR_sum/max_len
        A = A.cpu().numpy()
        Q = rwr_torch(A, 0.5)
        Q = torch.from_numpy(Q).to(device)
        R = torch.log(Q + 1 / ngene)
        RR_sum = torch.mm(R.T, R)
        del(Q)

    RR_sum = RR_sum.cpu().numpy()
    print(time.time()-s)
    if separate is None:
        x = network_svd(ndim, num_thread*torch_thread, RR_sum)
        del(RR_sum)
    else:
        x = np.concatenate(xs, axis=0)
        del(xs)
    return x
