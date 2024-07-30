import gdown
import json
import os
import zipfile

import pandas as pd

from .utils import DataHubBase


class FrcSub(DataHubBase):
    def __init__(self, root: str = "./tmp", **kwargs):
        super().__init__(**kwargs)
        # download FrcSub dataset from Google driver
        url = "https://drive.google.com/uc?id=1XLZFr-q_qqMjHW1MCBiIAwZr033o9s0x"
        filename = "Fraction_Sub_data.zip"
        os.path.expanduser(root)
        archive = os.path.join(root, filename)
        gdown.cached_download(url, archive)
        # load file from tmp folder
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            with zip_ref.open("config.json") as file:
                self.config = json.load(file)
            for file_name, file_path in self.config["files"].items():
                read_func = lambda path: pd.read_csv(zip_ref.open(path), header=None).to_numpy(dtype=float)
                exec("self.{} = {}".format(file_name, "read_func(\"{}\")".format(file_path)))
            for info_name, info_var in self.config["info"].items():
                exec("self.{} = {}".format(info_name, info_var))

        self._set_type_map["total"] = self.response
