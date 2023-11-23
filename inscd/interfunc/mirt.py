import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .._base import _InteractionFunction


class MIRT_IF(_InteractionFunction, nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

    @staticmethod
    def irt2pl(theta, a, b, F=torch):
        return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        knowledge_ts = kwargs["knowledge_ts"]
        q_mask = kwargs["q_mask"]
        if student_ts.shape[1] == knowledge_ts.shape[0]:
            return self.irt2pl(torch.sigmoid(student_ts) * q_mask, torch.sigmoid(diff_ts) * q_mask, torch.sigmoid(disc_ts).view(-1))
        else:
            return self.irt2pl(torch.sigmoid(student_ts), torch.sigmoid(diff_ts), torch.sigmoid(disc_ts).view(-1))


    def transform(self, mastery, knowledge):
        return F.sigmoid(mastery)
