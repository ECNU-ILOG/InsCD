import torch
import torch.nn as nn
import torch.nn.functional as F
from .._base import _InteractionFunction


class IRT_IF(_InteractionFunction, nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        q_mask = kwargs["q_mask"]
        input_x = torch.sigmoid(torch.sigmoid(disc_ts) * torch.sum((torch.sigmoid(student_ts) - torch.sigmoid(diff_ts)) * q_mask))
        return input_x.view(-1)

    def transform(self, mastery, knowledge):
        return F.sigmoid(mastery)
