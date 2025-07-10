import torch
import torch.nn as nn

from .._base import _InteractionFunction


class IRT_IF(_InteractionFunction, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.get('latent_num', config['knowledge_num'])
        if self.latent_dim is not None:
            self.transform_student = nn.Linear(self.latent_dim, 1)
            self.transform_exercise = nn.Linear(self.latent_dim, 1)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        if self.latent_dim is not None:
            input_x = torch.sigmoid(self.transform_student(student_ts) - self.transform_exercise(diff_ts))
        else:
            input_x = torch.sigmoid(student_ts - diff_ts)
        return input_x.view(-1)

    def transform(self, mastery, knowledge):
        return torch.sigmoid(mastery)
