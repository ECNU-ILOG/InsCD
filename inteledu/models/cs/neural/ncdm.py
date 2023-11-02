import torch
import torch.nn as nn
import torch.optim as optim

from ...._base import _CognitiveDiagnosisModel
from ....datahub import DataHub
from ....interfunc import NCD_IF

class NCDM(_CognitiveDiagnosisModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int):
        """
        Description:
        NCDM ...

        Parameters:
        student_num: int type
            The number of students in the response logs
        exercise_num: int type
            The number of exercises in the response logs
        knowledge_num: int type
            The number of knowledge concepts in the response logs
        method: Ignored
            Not used, present here for API consistency by convention.
        """
        super().__init__(student_num, exercise_num, knowledge_num)
    def build(self, hidden_dims: list=None, dropout=0.5, device="cpu", dtype=torch.float32, **kwargs):
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.inter_func = NCD_IF(student_num=self.student_num,
                                 exercise_num=self.exercise_num,
                                 knowledge_num=self.knowledge_num,
                                 hidden_dims=hidden_dims,
                                 dropout=dropout,
                                 device=device,
                                 dtype=dtype)

    def train(self, datahub: DataHub, set_type="train", valid_set_type="valid",
              valid_metrics=None, epoch=10, lr=0.01, weight_decay=0.0005, batch_size=256):
        if valid_metrics is None:
            valid_metrics = ["acc", "auc", "f1", "doa"]
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(self.inter_func.parameters(),lr=lr, weight_decay=weight_decay)
        for epoch_i in range(0, epoch):
            self._train(datahub=datahub, set_type=set_type,
                         valid_set_type=valid_set_type, valid_metrics=valid_metrics,
                         batch_size=256, loss_func=loss_func, optimizer=optimizer)

    def predict(self, datahub: DataHub, set_type, batch_size=256):
        return self._predict(datahub=datahub, set_type=set_type, batch_size=batch_size)

    def score(self, datahub: DataHub, set_type, metrics: list) -> dict:
        if metrics is None:
            metrics = ["acc", "auc", "f1", "doa"]
        return self._score(datahub=datahub, set_type=set_type, metrics=metrics)

    def diagnose(self):
        if self.inter_func is ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func["mastery"]

    def load(self, path: str):
        if self.inter_func is ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        self.inter_func.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.inter_func.state_dict(), path)
