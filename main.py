from data import DataReader
from dilate.dilate import dilate_loss
import torch


if __name__ == "__main__":

    dr = DataReader()
    df = dr.train.query("id == 1")

    windows = []
    window_size = 144
    for i in range(df.shape[0]):
        windows.append(df.iloc[i : i + window_size, -1].values)

    loss = dilate_loss(alpha=0.5, gamma=0.001)
    mse = torch.nn.MSELoss()

    batch = 1

    for i, _ in enumerate(windows):
        for j, _ in enumerate(windows):
            i = (
                torch.tensor(windows[i : i + batch], requires_grad=True)
                .to("cuda")
                .view(batch, -1, 1)
            )
            j = (
                torch.tensor(windows[j : j + batch], requires_grad=True)
                .to("cuda")
                .view(batch, -1, 1)
            )
            l = loss(i, j)
            # mse(i, j)
            # l.backward()
            break
        break
