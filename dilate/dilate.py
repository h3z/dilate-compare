from dilate import soft_dtw, path_soft_dtw
import torch, pickle
from datetime import datetime
import numpy as np


class TMP:
    def __init__(self) -> None:
        self.aaa = datetime.now()

    def ppp(self, i):
        print(i, datetime.now() - self.aaa)
        self.aaa = datetime.now()


tmp = TMP()
# def dilate_loss(outputs, targets, alpha=0.5, gamma=0.001):
def dilate_loss(items, alpha=0.5, gamma=0.001):
    start = datetime.now()
    # outputs, targets: shape (batch_size, N_output, 1)
    N = len(items[0])

    loss_shape = 0

    losses = np.zeros((len(items), len(items)))
    for i in range(len(items)):
        start = datetime.now()
        tmp.ppp(f"{i}-> start")

        paths = path_soft_dtw.compute_dilate_path(gamma, items, len(items) - i - 1)

        tmp.ppp(f"{i}-> compute 1 \t")
        # D = pickle.load(open("D.pkl", "rb"))
        loss_shapes = soft_dtw.compute_soft_dtw_batch(gamma, items, len(items) - i - 1)
        tmp.ppp(f"{i}-> compute 2 \t")

        for j in range(i + 1, len(items)):
            idx = j - i - 1

            loss_shape = loss_shapes[idx]
            path = paths[idx]

            omega = np.array(range(1, N + 1)).reshape(N, 1)
            Omega = soft_dtw.pairwise_distances(omega)
            loss_temporal = np.sum(path * Omega) / (N * N)

            loss = alpha * loss_shape + (1 - alpha) * loss_temporal
            # return loss, loss_shape, loss_temporal
            losses[i, j] = loss

        tmp.ppp(f"{i}-> add loss \t")
        print("finish: ", datetime.now() - start)
        print("time", datetime.now() - start)
