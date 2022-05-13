import pickle
import numpy as np
from data.data import DataReader
from base.dilate.dilate import dilate_loss
import torch
from datetime import datetime

if __name__ == "__main__":

    dr = DataReader()
    df = dr.train.query("id == 1")

    # windows = []
    # window_size = 144
    # for i in range(df.shape[0] - window_size):
    #     windows.append(df.iloc[i : i + window_size, -1].values)

    windows = pickle.load(open("data/input.pkl", "rb"))
    for w in windows:
        w[np.isnan(w)] = 0

    loss = dilate_loss(alpha=0.5, gamma=0.001)
    mse = torch.nn.MSELoss()

    batch = 1

    loss_values = []
    loss_shapes = []
    paths = []
    start = datetime.now()
    for i, _ in enumerate(windows):
        for j, _ in enumerate(windows[i + 1 :][:10]):
            print(j)
            item_i = (
                torch.tensor(windows[i : i + batch], requires_grad=True)
                .to("cuda")
                .view(batch, -1, 1)
            )
            item_j = (
                torch.tensor(windows[j + 1 : j + 1 + batch], requires_grad=True)
                .to("cuda")
                .view(batch, -1, 1)
            )
            loss_value, path, loss_shape = loss(item_i, item_j)
            loss_shapes.append(loss_shape.item())
            paths.append(path.detach().numpy())
            loss_values.append(loss_value.item())
            # mse(i, j)
            # l.backward()
        break

    print("time", datetime.now() - start)
    print("time")
