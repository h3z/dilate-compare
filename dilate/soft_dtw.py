import numpy as np
import torch, math
from numba import jit, cuda
from torch.autograd import Function

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


@cuda.jit
def compute_softdtw2(items, R, losses, batch_size):
    x = cuda.grid(1)

    item_i_idx = len(items) - batch_size - 1
    item_j_idx = item_i_idx + 1 + x

    item_i = items[item_i_idx]
    item_j = items[item_j_idx]
    if item_j_idx >= len(items):
        return

    R = R[x]
    R[0, 0] = 0.0
    N = len(items[0])

    for j in range(1, N + 1):
        for i in range(1, N + 1):
            r0 = -R[i - 1, j - 1] / gamma
            r1 = -R[i - 1, j] / gamma
            r2 = -R[i, j - 1] / gamma
            rmax = max(max(r0, r1), r2)
            rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
            softmin = -1.0 * gamma * (math.log(rsum) + rmax)
            R[i, j] = (item_i[i - 1] - item_j[j - 1]) ** 2 + softmin
    losses[x] = R[-2, -2]


def compute_soft_dtw_batch(gamma=0.001, items=None, batch_size=None):
    N = len(items[0])

    R = np.zeros((batch_size, N + 2, N + 2)) + 1e8
    losses = np.zeros(batch_size)
    dR = cuda.to_device(R)
    compute_softdtw2[batch_size // 1024 + 1, 1024](
        cuda.to_device(items), dR, losses, batch_size
    )
    return losses
