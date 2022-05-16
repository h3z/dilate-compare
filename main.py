from data.data import DataReader
from dilate.dilate import dilate_loss
import torch, pickle
import numpy as np
import warnings
import warnings

warnings.filterwarnings("ignore")
if __name__ == "__main__":

    # dr = DataReader()
    # df = dr.train.query("TurbID == 1")

    # windows = []
    # window_size = 144
    # for i in range(df.shape[0] - window_size):
    #     windows.append(df.iloc[i : i + window_size, -1].values)
    windows = pickle.load(open("data/input.pkl", "rb"))
    for w in windows:
        w[np.isnan(w)] = 0

    dilate_loss(windows)

    test_windws = [w[:5] for w in windows]
    dilate_loss(test_windws)
