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


def cuda_compute(vs, next_vs, next_qs):
    i = 0

    # temp = cuda.local.array((3), dtype=numba.float32)
    temp = [0] * 3
    temp[0] = vs[i + 0]
    temp[1] = vs[i + 1]
    temp[2] = vs[i + 2]
    my_min_x = temp
    # my min
    my_min_x[0] = -my_min_x[0]
    my_min_x[1] = -my_min_x[1]
    my_min_x[2] = -my_min_x[2]
    # use the log-sum-exp trick
    my_min_max_x = max(my_min_x[0], my_min_x[1], my_min_x[2])
    my_min_arr = [0] * 3
    # my_min_arr = cuda.local.array((3), dtype=nb.float32)
    my_min_arr[0] = math.exp((my_min_x[0] - my_min_max_x) / gamma)
    my_min_arr[1] = math.exp((my_min_x[1] - my_min_max_x) / gamma)
    my_min_arr[2] = math.exp((my_min_x[2] - my_min_max_x) / gamma)
    # exp_x = np.exp((x - max_x) / gamma)
    my_min_Z = my_min_arr[0] + my_min_arr[1] + my_min_arr[2]
    my_min_arr[0] /= my_min_Z
    my_min_arr[1] /= my_min_Z
    my_min_arr[2] /= my_min_Z
    v = -(gamma * math.log(my_min_Z) + my_min_max_x)
    next_qs[i, 0] = my_min_arr[0]
    next_qs[i, 1] = my_min_arr[1]
    next_qs[i, 2] = my_min_arr[2]

    # v, Q[i, j] = my_min(temp, gamma)
    # v = 1

    next_vs[i] = v


def dtw_grad2(theta, gamma=0.001):
    m = theta.shape[0]
    n = theta.shape[1]
    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    N = m + 1
    for i in range(1, N):
        vs = []
        for j in range(0, i):
            # print(f"({i-j}, {j})", end=" ")
            vs.append(V[i - j, j])
            vs.append(V[i - j - 1, j])
        vs.append(V[i - j - 1, j + 1])

        next_vs = [0] * i
        next_qs = np.array([[0, 0, 0] for i in range(i)])
        cuda_compute(vs, next_vs, next_qs)

        for j in range(i):
            # print(f"({i+1-j:2d}, {j:2d}),", end=" ")
            Q[i - j, j + 1] = next_qs[j]
            V[i - j, j + 1] = next_vs[j] + theta[i - j - 1, j]

    for i in range(N, 2 * N - 2):
        vs = []
        for j in range(i - N + 1, N - 1):
            # print(f"({i-j:2d}, {j:2d}), ({i-j-1:2d}, {j:2d})", end=" ")
            vs.append(V[i - j, j])
            vs.append(V[i - j - 1, j])
        # print(f"({i-j-1:2d}, {j+1:2d})")
        vs.append(V[i - j - 1, j + 1])

        next_vs = [0] * (2 * N - i - 2)
        next_qs = np.array([[0, 0, 0] for i in range(2 * N - i - 2)])
        cuda_compute(vs, next_vs, next_qs)

        for j in range(i - N + 1, N - 1):
            # print(j, end=" ")
            # print(f"({i-j:2d}, {j+1:2d}),", end=" ")
            Q[i - j, j + 1] = next_qs[j - i + N - 1]
            V[i - j, j + 1] = next_vs[j - i + N - 1] + theta[i - j - 1, j]

    # for i in range(1, m + 1):
    #     for j in range(1, n + 1):
    #         # theta is indexed starting from 0.
    #         v, Q[i, j] = my_min(
    #             np.array([V[i, j - 1], V[i - 1, j - 1], V[i - 1, j]]), gamma
    #         )
    #         V[i, j] = theta[i - 1, j - 1] + v

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


# @jit(nopython=True)
def my_max(x, gamma):
    # use the log-sum-exp trick
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z


# @jit(nopython=True)
def my_min(x, gamma):
    min_x, argmax_x = my_max(-x, gamma)
    return -min_x, argmax_x


# @jit(nopython=True)
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


if __name__ == "__main__":
    data = pickle.load(open("dump/dcpu.dump", "rb"))

    test1(data, 1)
    N = 10000
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
