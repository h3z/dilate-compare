from data.data import DataReader
from dilate.dilate import dilate_loss
import torch, pickle
import numpy as np
import warnings
import argparse

warnings.filterwarnings("ignore")
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    args = parser.parse_args()

    windows = pickle.load(open("data/input.pkl", "rb"))
    for w in windows:
        w[np.isnan(w)] = 0

    dilate_loss(windows, args.n)
