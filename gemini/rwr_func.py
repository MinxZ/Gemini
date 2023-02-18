import random
import sys

import numpy as np
import torch

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


def rwr_torch(A=None, restart_prob=None):
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    n = A.shape[0]
    a = (torch.eye(n) - (1 - restart_prob) * A).type(torch.Tensor)
    b = (restart_prob * torch.eye(n)).type(torch.Tensor)
    Q = torch.linalg.solve(a, b).cpu().numpy()
    return Q.T


def rwr(A=None, restart_prob=None):
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    n = A.shape[0]
    a = (np.eye(n) - (1 - restart_prob) * A).astype('float32')
    b = (restart_prob * np.eye(n)).astype('float32')
    # a = (np.eye(n) - (1 - restart_prob) * A)
    # b = (restart_prob * np.eye(n))
    Q = np.linalg.solve(a, b)
    return Q.T


def rwr_torch_iterative(A=None, restart_prob=None, delta_=1e-3, max_iter=10,
               verbal=True, device=None):
    torch.manual_seed(1)
    nnode, nfeat = A.shape
    for i in range(nnode):
        s = np.sum(A[i, :])
        if s > 0:
            A[i, :] = A[i, :] / s
        else:
            A[i, :] = 1. / nfeat
    # print(np.shape(A))
    # device = torch.device("cuda:0")
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif type(device) == str:
        device = torch.device(device)

    # print(device)
    nnode = A.shape[0]
    reset = np.eye(nnode)
    nsample, nnode = reset.shape
    P = A
    # P = P.T
    # norm_reset = reset
    # = norm_reset.astype(np.float16)
    norm_reset = torch.eye(nnode, dtype=torch.float16).to(device)
    P = P.astype(np.float16)
    # Q = torch.from_numpy(norm_reset).to(device)
    # norm_reset = torch.from_numpy(norm_reset).to(device)
    P = torch.from_numpy(P).to(device)  # .float()
    Q = norm_reset
    delta_min = 1.2
    cot = 0
    # t = time.time()
    for i in range(1, max_iter):
        # Q = gnp.garray(Q)
        # P = gnp.garray(P)
        Q_new = (1-restart_prob) * torch.mm(Q, P) + restart_prob*norm_reset
        # print(Q)
        delta = torch.norm(Q-Q_new, 2)
        Q = Q_new
        if verbal and i > 1:
            print('random walk', i, delta)
        # print 'random walk iter',i, delta
        sys.stdout.flush()
        if delta <= delta_:
            break
        if delta <= delta_min+1e-4:
            delta_min = delta
            cot = 1
        else:
            cot += 1
        if cot > 3:
            break
    # print('1')
    # Q = Q.cpu().numpy()
    # Q = Q.astype(np.float32)
    # print('0')
    return Q.T
