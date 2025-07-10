import torch
import torch.nn as nn
import torch.optim as optim

from ..._base import _CognitiveDiagnosisModel
from ...interfunc import CDMFKC_IF
from ...extractor import CDMFKC_EX


class CDMFKC(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        Cognitive Diagnosis Model Focusing on Knowledge Concepts (CDMFKC)
        Sheng Li et al. Cognitive Diagnosis Focusing on Knowledge Concepts. CIKM'22
        """
        super().__init__(config=config)
        self.extractor = CDMFKC_EX(config)
        self.inter_func = CDMFKC_IF(config)



    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])


    def get_attribute(self, attribute_name):
        if attribute_name == 'mastery':
            return self.diagnose().detach().cpu().numpy()
        elif attribute_name == 'diff':
            return self.inter_func.transform(self.extractor["diff"],
                                             self.extractor["knowledge"]).detach().cpu().numpy()
        elif attribute_name == 'knowledge':
            return self.extractor["knowledge"].detach().cpu().numpy()
        else:
            return None
