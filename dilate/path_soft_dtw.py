import numpy as np
import math
from numba import jit, cuda
import numba as nb
from datetime import datetime

TIMESTEPS = 144
gamma = 0.001


@cuda.jit
# def cuda_compute2(V, Q, l, theta, iiii):
def cuda_compute2(V, Q, l, theta, part, fix):
    # bi = cuda.blockIdx.x
    bi = 0
    iiii = cuda.threadIdx.x + fix

    if iiii >= l:
        cuda.syncthreads()
        return

    V = V[bi]
    Q = Q[bi]
    theta = theta[bi]

    if part == 1:
        y = iiii + 1
        x = l - y + 1
    elif part == 2:
        x = TIMESTEPS + 1 - l + iiii
        y = 2 * TIMESTEPS - l - x + 1

    arr0 = -V[x, y - 1]
    arr1 = -V[x - 1, y - 1]
    arr2 = -V[x - 1, y]
    # my min
    # use the log-sum-exp trick
    my_min_max_x = max(arr0, arr1, arr2)
    # my_min_arr = [0] * 3
    arr0 = math.exp((arr0 - my_min_max_x) / gamma)
    arr1 = math.exp((arr1 - my_min_max_x) / gamma)
    arr2 = math.exp((arr2 - my_min_max_x) / gamma)
    # exp_x = np.exp((x - max_x) / gamma)
    my_min_Z = arr0 + arr1 + arr2
    arr0 /= my_min_Z
    arr1 /= my_min_Z
    arr2 /= my_min_Z
    v = -(gamma * math.log(my_min_Z) + my_min_max_x)

    Q[x, y, 0] = arr0
    Q[x, y, 1] = arr1
    Q[x, y, 2] = arr2

    V[x, y] = v + theta[x - 1, y - 1]

    cuda.syncthreads()


@cuda.jit
def cuda_E(Q, E, batch, N):
    x = cuda.grid(1)
    if x >= batch:
        return

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


def compute_dilate_path(theta, gamma=0.001):
    # GRID_SIZE = 400_000
    # GRID_SIZE = 1
    while True:
        tmp.ppp(1)
        start = datetime.now()
        batch = theta.shape[0]

        N = theta.shape[1] + 1
        Q = np.zeros((batch, N + 1, N + 1, 3))
        V = np.zeros((batch, N, N))
        V[:, :, 0] = 1e10
        V[:, 0, :] = 1e10
        V[:, 0, 0] = 0

        tmp.ppp()
        dV = cuda.to_device(V)
        print(dV.alloc_size / 1024 / 1024)
        tmp.ppp(8)
        dQ = cuda.to_device(Q)
        print(dQ.alloc_size / 1024 / 1024)
        tmp.ppp(9)
        dtheta = cuda.to_device(theta)
        print(dtheta.alloc_size / 1024 / 1024)
        tmp.ppp(10)

        print("cp mem", datetime.now() - start)
        continue

        s = 1024

        for i in range(1, N):
            for fix in range(i // s + 1):
                cuda_compute2[batch, s](dV, dQ, i, dtheta, 1, fix * s)
            nb.cuda.synchronize()

        nb.cuda.synchronize()

        for i in range(N - 1, 0, -1):
            for fix in range(i // s + 1):
                cuda_compute2[batch, s](dV, dQ, i, dtheta, 2, fix * s)

        # V = dV.copy_to_host()
        # Q = dQ.copy_to_host()
        nb.cuda.synchronize()

        E = np.zeros((batch, N + 1, N + 1))
        E[:, N, :] = 0
        E[:, :, N] = 0
        E[:, N, N] = 1

        dE = cuda.to_device(E)

        cuda_E[batch // 1024 + 1, 1024](dQ, dE, batch, N)
        E = dE.copy_to_host()

        print(datetime.now() - start)

        # break

    return E[:, 1:N, 1:N]
