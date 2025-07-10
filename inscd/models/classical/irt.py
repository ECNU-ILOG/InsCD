import torch
import torch.nn as nn
import torch.optim as optim

from ..._base import _CognitiveDiagnosisModel
from ...interfunc import IRT_IF
from ...extractor import Default


class IRT(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        Multidimentional Item Response Theory (MIRT)
        Reference paper: Mark D. Reckase. Multidimensional Item Response Theory Models.
        """
        super().__init__(config=config)
        self.extractor = Default(config)
        self.inter_func = IRT_IF(config)



    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])
