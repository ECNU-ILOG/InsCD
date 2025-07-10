import torch
import torch.nn as nn
import torch.nn.functional as F
from .._base import _InteractionFunction


class MIRT_IF(_InteractionFunction, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['latent_dim']

    @staticmethod
    def irt2pl(theta, a, b, F=torch):
        return (1 / (1 + F.exp(- F.sum(F.multiply(a, theta), dim=-1) + b))).view(-1)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        student_ts = torch.squeeze(student_ts, dim=-1)
        diff_ts = torch.squeeze(diff_ts, dim=-1)
        disc_ts = torch.squeeze(disc_ts, dim=-1)
        return self.irt2pl(student_ts, F.softplus(diff_ts), disc_ts)

    def transform(self, mastery, knowledge):
        return torch.sigmoid(mastery)
