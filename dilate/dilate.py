from dilate import soft_dtw, path_soft_dtw
import torch, pickle
from datetime import datetime
import numpy as np
from numba import cuda
import numba as nb
from tqdm.auto import tqdm
from settings import FLOAT_TYPE

alpha = 0.5
# alpha = 0
gamma = 0.001


class TMP:
    def __init__(self) -> None:
        self.aaa = datetime.now()

    def ppp(self, i=None):
        if i:
            print(i, datetime.now() - self.aaa)
        self.aaa = datetime.now()


@cuda.jit
def loss_add(loss_shapes, paths, losses, omega, alpha):
    idx = cuda.grid(1)
    if idx >= len(loss_shapes):
        return
    N = len(omega)
    loss_shape = loss_shapes[idx]
    path = paths[idx][1:-1, 1:-1]

    sum = 0
    for i in range(len(path)):
        for j in range(len(path[0])):
            sum += path[i, j] * omega[i, j]

    loss_temporal = sum / N / N
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    losses[idx] = loss
    # losses[idx] = loss_temporal


tmp = TMP()
# def dilate_loss(outputs, targets, alpha=0.5, gamma=0.001):
def dilate_loss(items, worker_n):
    start = datetime.now()
    # outputs, targets: shape (batch_size, N_output, 1)
    # 按奇数处理，剩下的一个不要了。
    items = items[:-1]
    L = len(items)
    N = len(items[0])

    losses = np.zeros((len(items), len(items)))
    omega = np.array(range(1, N + 1)).reshape(N, 1)
    omega = soft_dtw.pairwise_distances(omega)
    domega = cuda.to_device(omega)

    items = np.array(items).astype(FLOAT_TYPE)
    ditems = nb.cuda.to_device(items)
    if worker_n is None:
        worker_n = 0

    n_gpu = 7
    # for row in range(worker_n, worker_n + 1):
    for row in tqdm(range(worker_n, (len(items) - 1) // 2, n_gpu)):
        start = datetime.now()

        # 前边部分是第 row 行。后边部分是第 L - row 行
        paths = path_soft_dtw.compute_dilate_path(gamma, ditems, row)
        assert paths.shape[0] == L

        loss_shapes = soft_dtw.compute_soft_dtw_batch(gamma, ditems, row)
        assert loss_shapes.shape[0] == L

        dloss = nb.cuda.device_array(shape=(L), dtype=FLOAT_TYPE)
        loss_add[L // 512 + 1, 512](loss_shapes, paths, dloss, domega, alpha)
        losses_rows = dloss.copy_to_host()

        for j in range(L - row - 1):
            losses[row, j + row + 1] = losses_rows[j]
        for j in range(L - row - 1, L):
            losses[L - row - 2, j] = losses_rows[j]

        # print(f"-> {str(datetime.now() - start)[6:]}")

    # pickle.dump(losses, open(f"losses_0517_{worker_n}.pkl", "wb"))
    return losses


def dilate_loss_one(items, row, alpha):
    items = items[:-1]
    L = len(items)
    N = len(items[0])

    losses = np.zeros((len(items), len(items)))
    omega = np.array(range(1, N + 1)).reshape(N, 1)
    omega = soft_dtw.pairwise_distances(omega)
    domega = cuda.to_device(omega)

    items = np.array(items).astype(FLOAT_TYPE)
    ditems = nb.cuda.to_device(items)

    paths = path_soft_dtw.compute_dilate_path(gamma, ditems, row)
    assert paths.shape[0] == L

    loss_shapes = soft_dtw.compute_soft_dtw_batch(gamma, ditems, row)
    assert loss_shapes.shape[0] == L

    dloss = nb.cuda.device_array(shape=(L), dtype=FLOAT_TYPE)
    loss_add[L // 512 + 1, 512](loss_shapes, paths, dloss, domega, alpha)

    losses_rows = dloss.copy_to_host()

    for j in range(L - row - 1):
        losses[row, j + row + 1] = losses_rows[j]
    for j in range(L - row - 1, L):
        losses[L - row - 2, j] = losses_rows[j]

    return losses[row]
