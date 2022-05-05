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


class TMP:
    def __init__(self) -> None:
        self.aaa = datetime.now()

    def ppp(self, i):
        # print("\t", i, datetime.now() - self.aaa, datetime.now())
        self.aaa = datetime.now()


tmp = TMP()


@cuda.jit
def cuda_compute(vs, next_vs, next_qs):
    iiii = cuda.threadIdx.x
    # iiii = 0
    arr0 = -vs[iiii * 2]
    arr1 = -vs[iiii * 2 + 1]
    arr2 = -vs[iiii * 2 + 2]
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
    next_qs[iiii, 0] = arr0
    next_qs[iiii, 1] = arr1
    next_qs[iiii, 2] = arr2

    # v, Q[i, j] = my_min(temp, gamma)
    # v = 1

    next_vs[iiii] = v


# @cuda.jit
def cuda_compute2(V, Q, l, theta, iiii):
    # def cuda_compute2(V, Q, l, theta):
    # iiii = cuda.threadIdx.x

    y = iiii + 1
    x = l - y + 1

    # iiii = 0
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

    # next_vs[iiii] = v
    V[x, y] = v + theta[x - 1, y - 1]


def dtw_grad2(theta, gamma=0.001):
    # for _ in range(300):
    while True:
        GRID_SIZE = 1
        m = theta.shape[0]
        n = theta.shape[1]
        V = np.zeros((m + 1, n + 1))
        V[:, 0] = 1e10
        V[0, :] = 1e10
        V[0, 0] = 0

        Q = np.zeros((m + 2, n + 2, 3))

        N = m + 1
        total = datetime.now() - datetime.now()
        sub_total = datetime.now() - datetime.now()
        for i in range(1, N):
            start = datetime.now()
            vs = []
            for j in range(i):
                # print(f"({i-j}, {j})", end=" ")
                vs.append(V[i - j, j])
                vs.append(V[i - j - 1, j])
            vs.append(V[i - j - 1, j + 1])

            next_vs = np.zeros(i)
            next_qs = np.array([[0, 0, 0] for i in range(i)])
            sub_start = datetime.now()
            # cuda_compute[GRID_SIZE, len(next_vs)](np.array(vs), next_vs, next_qs)
            # cuda_compute2[GRID_SIZE, len(next_vs)](
            #     np.array(V), np.array(Q), len(next_vs), theta
            # )
            for iiii in range(len(next_vs)):
                cuda_compute2(V, np.array(Q), len(next_vs), theta, iiii)
            sub_total += datetime.now() - sub_start

            # for j in range(i):
            #     # print(f"({i+1-j:2d}, {j:2d}),", end=" ")
            #     Q[i - j, j + 1] = next_qs[j]
            #     V[i - j, j + 1] = next_vs[j] + theta[i - j - 1, j]

            total += datetime.now() - start

        print(f"---> {sub_total}, {sub_total / total:.3f}")
        for i in range(N, 2 * N - 2):
            vs = []
            for j in range(i - N + 1, N - 1):
                # print(f"({i-j:2d}, {j:2d}), ({i-j-1:2d}, {j:2d})", end=" ")
                vs.append(V[i - j, j])
                vs.append(V[i - j - 1, j])
            # print(f"({i-j-1:2d}, {j+1:2d})")
            vs.append(V[i - j - 1, j + 1])

            next_vs = np.zeros(2 * N - i - 2)
            next_qs = np.array([[0, 0, 0] for i in range(2 * N - i - 2)])
            cuda_compute[GRID_SIZE, len(next_vs)](np.array(vs), next_vs, next_qs)

            for j in range(i - N + 1, N - 1):
                # print(j, end=" ")
                # print(f"({i-j:2d}, {j+1:2d}),", end=" ")
                Q[i - j, j + 1] = next_qs[j - i + N - 1]
                V[i - j, j + 1] = next_vs[j - i + N - 1] + theta[i - j - 1, j]

        E = np.zeros((m + 2, n + 2))
        # E[m + 1, :] = 0
        # E[:, n + 1] = 0
        # E[m + 1, n + 1] = 1
        # Q[m + 1, n + 1] = 1

        # for i in range(m, 0, -1):
        #     for j in range(n, 0, -1):
        #         E[i, j] = (
        #             Q[i, j + 1, 0] * E[i, j + 1]
        #             + Q[i + 1, j + 1, 1] * E[i + 1, j + 1]
        #             + Q[i + 1, j, 2] * E[i + 1, j]
        #         )

        # return V[m, n], E[1 : m + 1, 1 : n + 1], Q, E


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
