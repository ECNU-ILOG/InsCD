import torch
import torch.nn as nn
import torch.optim as optim

from ..._base import _CognitiveDiagnosisModel
from ...interfunc import KSCD_IF
from ...extractor import Default


class KSCD(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        Knowledge-sensed Cognitive Diagnosis Model (KSCD)
        Haiping Ma et al. Knowledge-Sensed Cognitive Diagnosis for Intelligent Education Platforms. CIKM'22.
        """
        super().__init__(config=config)
        self.extractor = Default(config)
        self.inter_func = KSCD_IF(config)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])
