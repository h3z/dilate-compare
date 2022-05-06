from datetime import datetime
from dilate.dilate import dilate_loss_
import torch
import sys
import pickle

import multiprocessing
import numpy as np
from numba import jit, cuda
import numba as nb
import math

gamma = 0.001

TIMESTEPS = 144


class TMP:
    def __init__(self) -> None:
        self.aaa = datetime.now()

    def ppp(self, i=None):
        if i:
            print("\t", i, datetime.now() - self.aaa)
        self.aaa = datetime.now()


tmp = TMP()


@cuda.jit
# def cuda_compute2(V, Q, l, theta, iiii):
def cuda_compute2(V, Q, l, theta, part, fix):
    iiii = cuda.threadIdx.x + fix

    if iiii >= l:
        cuda.syncthreads()
        return

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


def dtw_grad2(theta, gamma=0.001):
    # for _ in range(300):
    while True:
        start = datetime.now()
        GRID_SIZE = 400_000
        # GRID_SIZE = 1
        m = theta.shape[0]
        n = theta.shape[1]
        V = np.zeros((m + 1, n + 1))
        V[:, 0] = 1e10
        V[0, :] = 1e10
        V[0, 0] = 0

        Q = np.zeros((m + 2, n + 2, 3))

        N = m + 1

        dV = cuda.to_device(V)
        dQ = cuda.to_device(Q)
        dtheta = cuda.to_device(theta)

        s = 128 + 32

        tmp.ppp()
        for i in range(1, N):
            for fix in range(i // s + 1):
                cuda_compute2[GRID_SIZE, s](dV, dQ, i, dtheta, 1, fix * s)
            nb.cuda.synchronize()

        nb.cuda.synchronize()
        tmp.ppp("1   end: ")

        tmp.ppp()
        for i in range(N - 1, 0, -1):
            for fix in range(i // s + 1):
                cuda_compute2[GRID_SIZE, s](dV, dQ, i, dtheta, 2, fix * s)
            # cuda_compute2[GRID_SIZE, i](dV, dQ, i, dtheta, 2)

        V = dV.copy_to_host()
        Q = dQ.copy_to_host()
        nb.cuda.synchronize()
        tmp.ppp("2   end: ")

        E = np.zeros((m + 2, n + 2))
        E[m + 1, :] = 0
        E[:, n + 1] = 0
        E[m + 1, n + 1] = 1
        Q[m + 1, n + 1] = 1

        for i in range(m, 0, -1):
            for j in range(n, 0, -1):
                E[i, j] = (
                    Q[i, j + 1, 0] * E[i, j + 1]
                    + Q[i + 1, j + 1, 1] * E[i + 1, j + 1]
                    + Q[i + 1, j, 2] * E[i + 1, j]
                )

        # return V[m, n], E[1 : m + 1, 1 : n + 1], Q, E
        # print(datetime.now() - start)


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print("%s cost time: %.3f s" % (func.__name__, time_spend))
        return result

    return func_wrapper


@jit(nopython=True)
def my_max(x, gamma):
    # use the log-sum-exp trick
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z


@jit(nopython=True)
def my_min(x, gamma):
    min_x, argmax_x = my_max(-x, gamma)
    return -min_x, argmax_x


@jit(nopython=True)
def dtw_grad(theta, gamma=0.001):
    m = theta.shape[0]
    n = theta.shape[1]
    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            v, Q[i, j] = my_min(
                np.array([V[i, j - 1], V[i - 1, j - 1], V[i - 1, j]]), gamma
            )
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, :] = 0
    E[:, n + 1] = 0
    E[m + 1, n + 1] = 1
    Q[m + 1, n + 1] = 1

    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            E[i, j] = (
                Q[i, j + 1, 0] * E[i, j + 1]
                + Q[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + Q[i + 1, j, 2] * E[i + 1, j]
            )

    return V[m, n], E[1 : m + 1, 1 : n + 1], Q, E


@timer
def test1(data, N):
    for i in range(N):
        dtw_grad(data)


@timer
def test2(data, N, pool):
    pool.map(dtw_grad, [data] * N)


@timer
def test3(data, N):
    for i in range(2):
        dtw_grad2(data)


if __name__ == "__main__":
    data = pickle.load(open("dump/dcpu.dump", "rb"))

    N = 1000
    print(f"test on {N} computations")
    print("test1")
    # for i in range(3):
    #     res1 = test1(data, N)

    print(
        """(1_000 computations)
test1 cost time: 6.535 s
test1 cost time: 5.337 s
test1 cost time: 5.327 s
            """
    )

    print("Pool 2")
    # with multiprocessing.Pool(2) as p:
    #     for i in range(3):
    #         res2 = test2(data, N, p)
    print(
        """(1_000 computations)
test2 cost time: 4.849 s
test2 cost time: 3.641 s
test2 cost time: 3.601 s
      """
    )

    print("Pool 20")
    # with multiprocessing.Pool(20) as p:
    #     for i in range(3):
    #         res2 = test2(data, N, p)
    print(
        """(10_000 computations)
Pool 20
test2 cost time: 12.789 s
test2 cost time: 12.438 s
test2 cost time: 11.400 s
          """
    )

    print("Pool 128")
    # with multiprocessing.Pool(128) as p:
    #     for i in range(3):
    #         res2 = test2(data, N, p)
    print(
        """(10_000 computations)
Pool 128
test2 cost time: 11.284 s
test2 cost time: 11.198 s
test2 cost time: 8.890 s
          """
    )

    print("Pool 222")
    # with multiprocessing.Pool(222) as p:
    #     for i in range(5):
    #         res2 = test2(data, N, p)
    print(
        """(10_000 computations)
test2 cost time: 11.754 s
test2 cost time: 11.810 s
test2 cost time: 8.393 s
test2 cost time: 8.918 s
test2 cost time: 8.565 s
        """
    )

    print("cuda jit")
    for i in range(50):
        res2 = test3(data, N)
