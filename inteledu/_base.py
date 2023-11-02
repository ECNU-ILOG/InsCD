from abc import abstractmethod

from . import listener
from . import ruler


class _InteractionFunction:
    @abstractmethod
    def fit(self, datahub, set_type, **kwargs):
        ...

    @abstractmethod
    def compute(self, datahub, set_type, **kwargs):
        ...

    @abstractmethod
    def __getitem__(self, item):
        ...


class _CognitiveDiagnosisModel:
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int):
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        # ellipse members
        self.method = ...
        self.device: str = ...
        self.inter_func: _InteractionFunction = ...

    def _train(self, datahub, set_type="train",
                valid_set_type=None, valid_metrics=None, **kwargs):
        if self.inter_func is ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        self.inter_func.fit(datahub, set_type, **kwargs)
        if valid_set_type is not None:
            self.score(datahub, valid_set_type, valid_metrics)

    def _predict(self, datahub, set_type: str, **kwargs):
        if self.inter_func is ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.compute(datahub, set_type, **kwargs)

    @listener
    def _score(self, datahub, set_type: str, metrics: list, **kwargs):
        if self.inter_func is ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        pred_r = self.inter_func.compute(datahub, set_type, **kwargs)
        return ruler(self, datahub, set_type, pred_r, metrics)

    @abstractmethod
    def build(self, *args, **kwargs):
        ...

    @abstractmethod
    def train(self, datahub, set_type, valid_set_type=None, valid_metrics=None, **kwargs):
        ...

    @abstractmethod
    def predict(self, datahub, set_type, **kwargs):
        ...

    @abstractmethod
    def score(self, datahub, set_type, metrics: list)->dict:
        ...

    @abstractmethod
    def diagnose(self):
        ...

    @abstractmethod
    def load(self, path: str):
        ...

    @abstractmethod
    def save(self, path: str):
        ...
