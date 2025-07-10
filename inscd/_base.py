from abc import abstractmethod

from . import listener
# from . import ruler
# from . import Unifier


class _Extractor:
    @abstractmethod
    def extract(self, **kwargs):
        ...

    @abstractmethod
    def __getitem__(self, item):
        ...


class _InteractionFunction:
    @abstractmethod
    def compute(self, **kwargs):
        ...

    @abstractmethod
    def transform(self, mastery, knowledge):
        ...

    def monotonicity(self):
        ...


import torch.nn as nn
class _CognitiveDiagnosisModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inter_func: _InteractionFunction = ...
        self.extractor: _Extractor = ...

    @abstractmethod
    def diagnose(self):
        ...

