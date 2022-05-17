from dilate import soft_dtw, path_soft_dtw
import torch, pickle
from datetime import datetime
import numpy as np
from numba import cuda
import numba as nb
from tqdm.auto import tqdm


class TMP:
    def __init__(self) -> None:
        self.aaa = datetime.now()

    def ppp(self, i=None):
        if i:
            print(i, datetime.now() - self.aaa)
        self.aaa = datetime.now()


@cuda.jit
def loss_add(loss_shapes, paths, losses, omega, N, alpha):
    idx = cuda.grid(1)
    if idx >= len(loss_shapes):
        return

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
def dilate_loss(items, worker_n, alpha=0.5, gamma=0.001):
    start = datetime.now()
    # outputs, targets: shape (batch_size, N_output, 1)
    N = len(items[0])

    losses = np.zeros((len(items), len(items)))
    omega = np.array(range(1, N + 1)).reshape(N, 1)
    omega = soft_dtw.pairwise_distances(omega)
    domega = cuda.to_device(omega)

    items = np.array(items).astype("float64")
    ditems = nb.cuda.to_device(items)
    worker_n = 0
    for i in tqdm(range(worker_n, len(items) - 1, 7)):
        start = datetime.now()
        tmp.ppp()

        batch_size = len(items) - i - 1

        paths = path_soft_dtw.compute_dilate_path(gamma, ditems, batch_size)
        tmp.ppp(f"{i} 1:")

        loss_shapes = soft_dtw.compute_soft_dtw_batch(gamma, ditems, batch_size)
        tmp.ppp(f"{i} 2:")
        dloss = nb.cuda.device_array(shape=(batch_size), dtype=np.float64)
        loss_add[batch_size // 1024 + 1, 1024](
            loss_shapes,
            paths,
            dloss,
            domega,
            N,
            0.5,
        )
        losses_i = dloss.copy_to_host()

        for j in range(batch_size):
            losses[i, j + i + 1] = losses_i[j]
        tmp.ppp(f"{i} 3:")
        print(f"-> {str(datetime.now() - start)[6:]}")

    pickle.dump(losses, open(f"/losses_0517_{worker_n}.pkl", "wb"))
