import pandas as pd
import json

from .utils import DataHubBase


class DataHub(DataHubBase):
    def __init__(self, data_source: str, **kwargs):
        super().__init__(**kwargs)
        # scan the datasets to load all available files
        if type(data_source) is str:
            with open(data_source + "/config.json") as file:
                self.config = json.load(file)
            for file_name, file_path in self.config["files"].items():
                read_func = lambda path: pd.read_csv(data_source + "/" + path, header=None).to_numpy(dtype=float)
                exec("self.{} = {}".format(file_name, "read_func(\"{}\")".format(file_path)))
            for info_name, info_var in self.config["info"].items():
                exec("self.{} = {}".format(info_name, info_var))
        else:
            raise ValueError("Unexpected type of \"data_source\" {}".format(type(data_source)))

        self._set_type_map["total"] = self.response
