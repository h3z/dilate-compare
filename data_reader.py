import pandas as pd


class DataReader:
    def __init__(self):
        DATA_ROOT = "/home/yanhuize/kdd2022/dataset/sdwpf/"
        turbine_datapath = f"{DATA_ROOT}/sdwpf_baidukddcup2022_full.pkl"
        location_datapath = f"{DATA_ROOT}/sdwpf_baidukddcup2022_turb_location.pkl"

        self.train = pd.read_pickle(turbine_datapath)
        self.location = pd.read_pickle(location_datapath)
