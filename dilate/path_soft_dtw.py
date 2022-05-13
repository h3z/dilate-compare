import numpy as np
import math
from numba import jit, cuda
import numba as nb
from datetime import datetime

TIMESTEPS = 144
gamma = 0.001


@cuda.jit
# def cuda_compute2(V, Q, l, theta, iiii):
def cuda_compute2(items, V, Q, l, part):
    iiii = cuda.threadIdx.x

    if iiii >= l:
        return
    if part == 1:
        y = iiii + 1
        x = l - y + 1
    elif part == 2:
        x = TIMESTEPS + 1 - l + iiii
        y = 2 * TIMESTEPS - l - x + 1

    bi = cuda.blockIdx.x
    Q = Q[bi]
    V = V[bi]
    V[1:, 0] = 1e10
    V[0, 1:] = 1e10

    item_i_idx = len(items) - cuda.gridDim.x - 1
    item_j_idx = item_i_idx + 1 + bi

    item_i = items[item_i_idx][y - 1]
    item_j = items[item_j_idx][x - 1]
    theta_v = (item_i - item_j) ** 2
    # theta_v = max(theta_v, 0.0)

    arr0 = -V[x, y - 1]
    arr1 = -V[x - 1, y - 1]
    arr2 = -V[x - 1, y]

    # V[x, y] = min(-arr0, -arr1, -arr2) + theta_v

    # my min
    # use the log-sum-exp trick
    my_min_max_x = max(arr0, arr1, arr2)
    # my_min_arr = [0] * 3
    arr0 = math.exp((arr0 - my_min_max_x) / gamma)
    arr1 = math.exp((arr1 - my_min_max_x) / gamma)
    arr2 = math.exp((arr2 - my_min_max_x) / gamma)
    # exp_x = np.exp((x - max_x) / gamma)
    my_min_Z = arr0 + arr1 + arr2
    v = -(gamma * math.log(my_min_Z) + my_min_max_x)

    Q[x, y, 0] = arr0 / my_min_Z
    Q[x, y, 1] = arr1 / my_min_Z
    Q[x, y, 2] = arr2 / my_min_Z

    # V[x, y] = v + theta[x - 1, y - 1]
    V[x, y] = v + theta_v


@cuda.jit
def cuda_E(Q, E, batch, N):
    x = cuda.grid(1)
    if x >= batch:
        return

    E[x, N, :] = 0
    E[x, :, N] = 0
    E[x, N, N] = 1

    Q[x, N, N] = 1
    for i in range(N - 1, 0, -1):
        for j in range(N - 1, 0, -1):
            E[x, i, j] = (
                Q[x, i, j + 1, 0] * E[x, i, j + 1]
                + Q[x, i + 1, j + 1, 1] * E[x, i + 1, j + 1]
                + Q[x, i + 1, j, 2] * E[x, i + 1, j]
            )


class TMP:
    def __init__(self) -> None:
        self.aaa = datetime.now()

    def ppp(self, i=None):
        if i:
            print(i, datetime.now() - self.aaa)
        self.aaa = datetime.now()


tmp = TMP()


def compute_dilate_path(gamma=0.001, items=None, batch=None):
    # GRID_SIZE = 400_000
    # GRID_SIZE = 1
    while True:
        N = len(items[0]) + 1

        items = np.array(items).astype("float64")
        ditems = nb.cuda.to_device(items)
        dV = nb.cuda.device_array(shape=(batch, N, N), dtype=np.float64)
        dQ = nb.cuda.device_array(shape=(batch, N + 1, N + 1, 3), dtype=np.float64)
        # dtheta = cuda.to_device(theta)

        for i in range(1, N):
            cuda_compute2[batch, i](ditems, dV, dQ, i, 1)

        nb.cuda.synchronize()

        for i in range(N - 1, 0, -1):
            cuda_compute2[batch, i](ditems, dV, dQ, i, 2)

        nb.cuda.synchronize()

        dE = nb.cuda.device_array(shape=(batch, N + 1, N + 1), dtype=np.float64)
        cuda_E[batch // 1024 + 1, 1024](dQ, dE, batch, N)
        E = dE.copy_to_host()
        break

    return E[:, 1:N, 1:N]
