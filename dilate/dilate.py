from dilate import soft_dtw
from dilate import path_soft_dtw
import torch
from datetime import datetime


class TMP:
    def __init__(self) -> None:
        self.aaa = datetime.now()

    def ppp(self, i):
        # print(i, datetime.now() - self.aaa, datetime.now())
        self.aaa = datetime.now()


def dilate_loss(alpha, gamma, device="cpu"):
    tmp = TMP()

    def dilate_loss_(outputs, targets):
        tmp.ppp(1)
        # outputs, targets: shape (batch_size, N_output, 1)
        batch_size, N_output = outputs.shape[0:2]
        loss_shape = 0
        softdtw_batch = soft_dtw.SoftDTWBatch.apply
        D = torch.zeros((batch_size, N_output, N_output)).to(device)
        for k in range(batch_size):
            tmp.ppp(2)
            Dk = soft_dtw.pairwise_distances(
                targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1)
            )
            tmp.ppp(3)
            D[k : k + 1, :, :] = Dk
        tmp.ppp(4)
        loss_shape = softdtw_batch(D, gamma)
        tmp.ppp(5)

        path_dtw = path_soft_dtw.PathDTWBatch.apply
        tmp.ppp(6)
        path = path_dtw(D, gamma)
        tmp.ppp(7)
        Omega = soft_dtw.pairwise_distances(
            torch.range(1, N_output).view(N_output, 1)
        ).to(device)
        tmp.ppp(8)
        loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
        loss = alpha * loss_shape + (1 - alpha) * loss_temporal
        # return loss, loss_shape, loss_temporal
        return loss

    return dilate_loss_


def dilate_loss_(outputs, targets, alpha=0.5, gamma=0.001, device="cpu"):
    tmp = TMP()
    tmp.ppp(1)
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]
    loss_shape = 0
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((batch_size, N_output, N_output)).to(device)
    for k in range(batch_size):
        tmp.ppp(2)
        Dk = soft_dtw.pairwise_distances(
            targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1)
        )
        tmp.ppp(3)
        D[k : k + 1, :, :] = Dk
    tmp.ppp(4)
    loss_shape = softdtw_batch(D, gamma)
    tmp.ppp(5)

    path_dtw = path_soft_dtw.PathDTWBatch.apply
    tmp.ppp(6)
    path = path_dtw(D, gamma)
    tmp.ppp(7)
    Omega = soft_dtw.pairwise_distances(torch.range(1, N_output).view(N_output, 1)).to(
        device
    )
    tmp.ppp(8)
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    # return loss, loss_shape, loss_temporal
    return loss
