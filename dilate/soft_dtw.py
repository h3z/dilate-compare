import numpy as np
import torch, math
from numba import jit, cuda
from torch.autograd import Function
import numba as nb
from settings import FLOAT_TYPE

gamma = 0.001


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x**2).sum(1).reshape(-1, 1)
    if y is not None:
        # y_t = torch.transpose(y, 0, 1)
        y_t = y.T
        y_norm = (y**2).sum(1).reshape(1, -1)
    else:
        y_t = x.T
        # torch.transpose(x, 0, 1)
        y_norm = x_norm.reshape(1, -1)

    # dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    dist = x_norm + y_norm - 2.0 * np.matmul(x, y_t)

    # return torch.clamp(dist, 0.0, float("inf"))
    return np.clip(dist, 0.0, float("inf"))


@cuda.jit(device=True)
def i_j_loss(R, item_i, item_j):
    N = len(item_i)

    for j in range(1, N + 1):
        for i in range(1, N + 1):
            r0 = -R[i - 1, j - 1] / gamma
            r1 = -R[i - 1, j] / gamma
            r2 = -R[i, j - 1] / gamma
            rmax = max(r0, r1, r2)
            rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
            softmin = -1.0 * gamma * (math.log(rsum) + rmax)
            R[i, j] = (item_i[i - 1] - item_j[j - 1]) ** 2 + softmin
    return R[-2, -2]


@cuda.jit
def compute_softdtw2(items, R, losses, row):
    idx = cuda.grid(1)
    L = items.shape[0]
    origin_idx = idx

    if idx >= L:
        return
    else:
        R = R[idx]
        R[0, 1:] = 1e8
        R[1:, 0] = 1e8
        R[0, 0] = 0.0

    size = L - row - 1
    if idx >= size:
        idx = idx - size
        row = L - row - 2

    item_i_idx = row
    item_j_idx = item_i_idx + 1 + idx
    item_j = items[item_j_idx]
    item_i = items[item_i_idx]

    losses[origin_idx] = i_j_loss(R, item_i, item_j)


def compute_soft_dtw_batch(gamma=0.001, ditems=None, row=None):
    L, N = ditems.shape

    # 这里都把 batch 改成 L，按L来。前边是row行的L-row-1个，后边是 倒数 row 行的那些 row +1 个，合计L个。
    dR = nb.cuda.device_array(shape=(L, N + 2, N + 2), dtype=FLOAT_TYPE)
    dlosses = nb.cuda.device_array(shape=(L), dtype=FLOAT_TYPE)
    compute_softdtw2[L // 512 + 1, 512](ditems, dR, dlosses, row)
    nb.cuda.synchronize()
    return dlosses
