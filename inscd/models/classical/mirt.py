import torch
import torch.nn as nn
import torch.optim as optim

from ..._base import _CognitiveDiagnosisModel
from ...interfunc import MIRT_IF
from ...extractor import Default


class MIRT(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        Multidimentional Item Response Theory (MIRT)
        Reference paper: Mark D. Reckase. Multidimensional Item Response Theory Models.
        """
        super().__init__(config=config)
        self.extractor = Default(config)
        self.inter_func = MIRT_IF(config)

    # def build(self, datahub, latent_dim: int, if_type='sum', device="cpu", dtype=torch.float64, **kwargs):
    #     self.student_num = datahub.student_num
    #     self.exercise_num = datahub.exercise_num
    #     self.knowledge_num = datahub.knowledge_num

    #     self.extractor = Default(
    #         student_num=self.student_num,
    #         exercise_num=self.exercise_num,
    #         knowledge_num=self.knowledge_num,
    #         device=device,
    #         dtype=dtype,
    #         latent_dim=latent_dim
    #     )
    #     if if_type == 'sub':
    #         self.inter_func = IRT_IF(
    #             device=device,
    #             dtype=dtype
    #         )
    #     elif if_type == 'sum':
    #         self.inter_func = MIRT_IF(
    #             knowledge_num=self.knowledge_num,
    #             latent_dim=latent_dim,
    #             device=device,
    #             dtype=dtype
    #         )
    #     else:
    #         raise ValueError('to be assigned')


    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])
