import numpy as np
import math
import numba
from numba import jit, cuda
import numba as nb
from datetime import datetime
import pickle

gamma = 0.001

CONSTANT_N = 145


@cuda.jit
def cuda_compute2(
    items,
    Q,
    batch_size,
):
    idx = cuda.grid(1)
    L, N = items.shape
    N += 1
    item_i_idx = L - batch_size - 1
    item_j_idx = item_i_idx + 1 + idx
    if item_j_idx >= L:
        return

    Q = Q[idx]
    V_pre = cuda.local.array(CONSTANT_N, nb.float64)
    V = cuda.local.array(CONSTANT_N, nb.float64)

    V_pre[1:] = 1e10
    V_pre[0] = 0
    V[0] = 1e10

    for x in range(1, N):
        item_j = items[item_j_idx][x - 1]
        for y in range(1, N):
            item_i = items[item_i_idx][y - 1]
            theta_v = (item_i - item_j) ** 2

            arr0 = -V[y - 1]
            arr1 = -V_pre[y - 1]
            arr2 = -V_pre[y]
            my_min_max_x = max(arr0, arr1, arr2)

            arr0 = math.exp((arr0 - my_min_max_x) / gamma)
            arr1 = math.exp((arr1 - my_min_max_x) / gamma)
            arr2 = math.exp((arr2 - my_min_max_x) / gamma)

            my_min_Z = arr0 + arr1 + arr2
            v = -(gamma * math.log(my_min_Z) + my_min_max_x)
            Q[x, y, 0] = arr0 / my_min_Z
            Q[x, y, 1] = arr1 / my_min_Z
            Q[x, y, 2] = arr2 / my_min_Z
            V[y] = v + theta_v

        for y in range(N):
            V_pre[y] = V[y]


@cuda.jit
def cuda_E(Q, E, batch, N):
    x = cuda.grid(1)
    if x >= batch:
        return

    E = E[x]
    E[N, :] = 0
    E[:, N] = 0
    E[N, N] = 1
    Q = Q[x]
    Q[N, N] = 1
    for i in range(N - 1, 0, -1):
        for j in range(N - 1, 0, -1):
            E[i, j] = (
                Q[i, j + 1, 0] * E[i, j + 1]
                + Q[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + Q[i + 1, j, 2] * E[i + 1, j]
            )


class TMP:
    def __init__(self) -> None:
        self.aaa = datetime.now()

    def ppp(self, i=None):
        if i:
            print(i, datetime.now() - self.aaa)
        self.aaa = datetime.now()


tmp = TMP()


def compute_dilate_path(gamma=0.001, ditems=None, batch=None):
    N = ditems.shape[1] + 1

    dQ = nb.cuda.device_array(shape=(batch, N + 1, N + 1, 3), dtype=np.float64)
    dE = nb.cuda.device_array(shape=(batch, N + 1, N + 1), dtype=np.float64)

    cuda_compute2[batch // 1024 + 1, 1024](ditems, dQ, batch)
    cuda_E[batch // 1024 + 1, 1024](dQ, dE, batch, N)

    return dE
