import warnings
import torch
import scipy.stats as ss
import numpy as np
import torch.nn as nn
from inscd._base import _CognitiveDiagnosisModel, _InteractionFunction
from inscd.datahub import DataHub


class NeuralDINA(_InteractionFunction, nn.Module):
    def __init__(self, student_num, exercise_num, knowledge_num, device, dtype, max_slip=0.4, max_guess=0.4):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.device = device
        self.dtype = dtype
        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess

        self.guess = nn.Embedding(self.exercise_num, 1)
        self.slip = nn.Embedding(self.exercise_num, 1)
        self.theta = nn.Embedding(self.student_num, self.knowledge_num)

    def forward(self, student_id, exercise_id, knowledge_id):
        theta = self.theta(student_id)
        slip = torch.squeeze(torch.sigmoid(self.slip(exercise_id)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(exercise_id)) * self.max_guess)
        if self.training:
            n = torch.sum(knowledge_id * (torch.sigmoid(theta) - 0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                               1e-6), self.step + 1 if self.step < self.max_step else 0
            return torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            n = torch.prod(knowledge_id * (theta >= 0) + (1 - knowledge_id), dim=1)
            return (1 - slip) ** n * guess ** (1 - n)

    def fit(self, response_logs_data, *args, **kwargs):
        pass

    def compute(self, response_logs_data):
        pass

    def __getitem__(self, item):
        pass

class DINA(_CognitiveDiagnosisModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, method: str):
        """
        Parameters:
        student_num: the number of students in the response logs
        exercise_num: the number of exercises in the response logs
        knowledge_num: the number of attributes
        method: the way to estimate parameters in this model, we provide ["neural", "em"]
        """
        super().__init__(student_num, exercise_num, knowledge_num, method)
        if self.method not in ["neural"]:
            raise ValueError("Do not support method {}. Currently we only support \"neural\", "
                             "\"em\" and \"grad\"".format(self.method))

    def build(self, device=None, dtype=torch.float32):
        if self.method == "neural" and device == "cuda" and not torch.cuda.is_available():
            warnings.warn("We find that you try to use \"neural\" method and \"cuda\" device to build DINA interaction "
                          "function, but \"cuda\" is not available. We have set it as \"cpu\".")

        if self.method == "neural" and device is None:
            warnings.warn("We find that you try to use \"neural\" method to build DINA interaction function but forget"
                          "pass parameter \"device\". We have set it as \"cpu\".")


    def train(self, response_logs: DataHub, set_type, valid_set_type=None, valid_metrics=None, *args, **kwargs):
        pass

    def predict(self, response_logs: DataHub, set_type):
        pass

    def score(self, response_logs: DataHub, set_type, metrics: list) -> dict:
        pass

    def diagnose(self):
        pass

    def load(self, path: str):
        pass

    def save(self, name: str, path: str):
        pass