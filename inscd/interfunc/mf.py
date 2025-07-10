import torch
import torch.nn as nn

from .._base import _InteractionFunction


class MF_IF(_InteractionFunction, nn.Module):
    def __init__(self, config):
        super().__init__()

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        q_mask = kwargs["q_mask"]
        input_x = torch.sigmoid(torch.sigmoid(disc_ts) * ((student_ts * q_mask) @ (diff_ts * q_mask).T))
        return input_x.view(-1)

    def transform(self, mastery, knowledge):
        return torch.sigmoid(mastery)
