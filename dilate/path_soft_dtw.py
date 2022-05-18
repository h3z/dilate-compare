import numpy as np
import math
import numba
from numba import jit, cuda
import numba as nb
from datetime import datetime
import pickle
from settings import FLOAT_TYPE

gamma = 0.001

CONSTANT_N = 145


@cuda.jit(device=True)
def i_j_path(Q, item_1, item_2):
    N = len(item_1) + 1

    V_pre = cuda.local.array(CONSTANT_N, "float64")
    V = cuda.local.array(CONSTANT_N, "float64")

    V_pre[1:] = 1e10
    V_pre[0] = 0
    V[0] = 1e10

    for x in range(1, N):
        item_j = item_1[x - 1]
        for y in range(1, N):
            item_i = item_2[y - 1]
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
def cuda_compute2(
    items,
    Q,
    row,
):
    idx = cuda.grid(1)
    L = items.shape[0]

    if idx >= L:
        return
    else:
        Q = Q[idx]

    size = L - row - 1
    if idx >= size:
        idx = idx - size
        row = L - row - 2

    item_i_idx = row
    item_j_idx = item_i_idx + 1 + idx
    item_1 = items[item_j_idx]
    item_2 = items[item_i_idx]

    i_j_path(Q, item_1, item_2)


@cuda.jit
def cuda_E(Q, E, N):
    x = cuda.grid(1)
    if x >= len(E):
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


def compute_dilate_path(gamma=0.001, ditems=None, row=None):
    L, N = ditems.shape

    # 这里都把 batch 改成 L，按L来。前边是row行的L-row-1个，后边是 倒数 row 行的那些 row +1 个，合计L个。
    dQ = nb.cuda.device_array(shape=(L, N + 2, N + 2, 3), dtype=FLOAT_TYPE)
    dE = nb.cuda.device_array(shape=(L, N + 2, N + 2), dtype=FLOAT_TYPE)
    cuda_compute2[L // 512 + 1, 512](ditems, dQ, row)
    nb.cuda.synchronize()

    cuda_E[L // 512 + 1, 512](dQ, dE, N + 1)
    nb.cuda.synchronize()

    return dE
