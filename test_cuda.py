from numba import cuda
import pickle
import numpy as np
import numba
import numba as nb
import math

gamma = 1e-3


@cuda.jit(
    nb.types.Tuple((nb.float32, nb.float32[:]))(nb.float32[:], nb.float32), device=True
)
def my_max(x, gamma):
    # use the log-sum-exp trick
    max_x = max(x[0], x[1], x[2])
    arr = cuda.local.array((3), dtype=nb.float32)
    arr[0] = math.exp((x[0] - max_x) / gamma)
    arr[1] = math.exp((x[1] - max_x) / gamma)
    arr[2] = math.exp((x[2] - max_x) / gamma)
    # exp_x = np.exp((x - max_x) / gamma)
    Z = arr[0] + arr[1] + arr[2]
    arr[0] /= Z
    arr[1] /= Z
    arr[2] /= Z
    return gamma * math.log(Z) + max_x, arr


# @cuda.jit(
#     nb.types.Tuple((nb.float32, nb.float32[:]))(nb.float32[:], nb.float32), device=True
# )
# def my_min(my_min_x, gamma):


def cuda_compute(vs, next_vs, next_qs):
    for iiii in range(len(next_vs)):

        # temp = cuda.local.array((3), dtype=numba.float32)
        temp = [0] * 3
        temp[0] = vs[iiii * 2]
        temp[1] = vs[iiii * 2 + 1]
        temp[2] = vs[iiii * 2 + 2]
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
        next_qs[iiii, 0] = my_min_arr[0]
        next_qs[iiii, 1] = my_min_arr[1]
        next_qs[iiii, 2] = my_min_arr[2]

        # v, Q[i, j] = my_min(temp, gamma)
        # v = 1

        next_vs[iiii] = v


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


@cuda.jit
def dtw_grad(theta):
    gamma = 0.001
    m = theta.shape[0]
    n = theta.shape[1]
    # V = np.zeros((m + 1, n + 1), dtype=np.float32)
    V = cuda.local.array((145, 145), dtype=numba.float32)
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = cuda.local.array((146, 146, 3), dtype=numba.float32)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            temp = cuda.local.array((3), dtype=numba.float32)
            temp[0] = V[i, j - 1]
            temp[1] = V[i - 1, j - 1]
            temp[2] = V[i - 1, j]

            my_min_x = temp
            # my min
            my_min_x[0] = -my_min_x[0]
            my_min_x[1] = -my_min_x[1]
            my_min_x[2] = -my_min_x[2]
            # use the log-sum-exp trick
            my_min_max_x = max(my_min_x[0], my_min_x[1], my_min_x[2])
            my_min_arr = cuda.local.array((3), dtype=nb.float32)
            my_min_arr[0] = math.exp((my_min_x[0] - my_min_max_x) / gamma)
            my_min_arr[1] = math.exp((my_min_x[1] - my_min_max_x) / gamma)
            my_min_arr[2] = math.exp((my_min_x[2] - my_min_max_x) / gamma)
            # exp_x = np.exp((x - max_x) / gamma)
            my_min_Z = my_min_arr[0] + my_min_arr[1] + my_min_arr[2]
            my_min_arr[0] /= my_min_Z
            my_min_arr[1] /= my_min_Z
            my_min_arr[2] /= my_min_Z
            v = -(gamma * math.log(my_min_Z) + my_min_max_x)
            Q[i, j, 0] = my_min_arr[0]
            Q[i, j, 1] = my_min_arr[1]
            Q[i, j, 2] = my_min_arr[2]

            # v, Q[i, j] = my_min(temp, gamma)
            # v = 1

            V[i, j] = theta[i - 1, j - 1] + v

    # E = cuda.local.array((146, 146), dtype=numba.float32)
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


@cuda.jit
def increment_by_one(an_array):
    x, y = cuda.grid(2)
    an_array[x, y] = x + y
    return
    # print(pos)
    # if pos < an_array.size:
    #     an_array[pos] += 1


if __name__ == "__main__":
    data = pickle.load(open("dump/dcpu.dump", "rb"))
    print(data.shape)
    # threadsperblock = 32
    # blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
    # increment_by_one[blockspergrid, threadsperblock](data)
    a = np.zeros((3, 4))
    increment_by_one[(3, 4), (1, 1)](a)
    print(a)

    dtw_grad[1, 1](data)
