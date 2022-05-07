import pandas as pd

to_custom_names = {
    "TurbID": "id",
    "Day": "day",
    "Tmstamp": "time",
    "Wspd": "spd",
    "Wdir": "dir",
    "Etmp": "environment_tmp",
    "Itmp": "inside_tmp",
    "Ndir": "nacelle_dir",
    "Pab1": "pab1",
    "Pab2": "pab2",
    "Pab3": "pab3",
    "Prtv": "reactive_power",
    "Patv": "active_power",
}


class DataReader:
    def __init__(self):
        DATA_ROOT = "/home/yanhuize/kdd2022/dataset/sdwpf/"
        turbine_datapath = f"{DATA_ROOT}/sdwpf_baidukddcup2022_full.pkl"
        location_datapath = f"{DATA_ROOT}/sdwpf_baidukddcup2022_turb_location.pkl"

        self.train = pd.read_pickle(turbine_datapath)
        self.location = pd.read_pickle(location_datapath)

        self.train = self.rename_columns(self.train)

    def rename_columns(self, df):
        return df.rename(columns=to_custom_names)

    @property
    def feature_cols(self):
        return [
            "spd",
            "dir",
            "environment_tmp",
            "inside_tmp",
            "nacelle_dir",
            "pab1",
            "pab2",
            "pab3",
            "reactive_power",
            "active_power",
        ]
