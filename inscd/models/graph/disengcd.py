import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from ..._base import _CognitiveDiagnosisModel
from ...interfunc import get_interaction_function
from ...extractor import DisenGCD_EX


class DisenGCD(_CognitiveDiagnosisModel):
    def __init__(self, config):
        super().__init__(config=config)
        self.extractor = DisenGCD_EX(config)
        inter_func_class, require_transfer = get_interaction_function(config.get('if_type','DisenGCD_IF'))
        self.inter_func = inter_func_class(config)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

