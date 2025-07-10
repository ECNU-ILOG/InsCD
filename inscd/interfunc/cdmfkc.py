import torch
import torch.nn as nn

from collections import OrderedDict

from ._util import none_neg_clipper
from .._base import _InteractionFunction


class CDMFKC_IF(_InteractionFunction, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.knowledge_num = config['knowledge_num']
        self.g_impact_a =  config['g_impact_a']
        self.g_impact_b =  config['g_impact_b']
        self.hidden_dims =  config['hidden_dims']
        self.dropout = config['dropout']

        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.knowledge_num, hidden_dim),
                        'activation0': nn.Tanh()
                    }
                )
            else:
                layers.update(
                    {
                        'dropout{}'.format(idx): nn.Dropout(p=self.dropout),
                        'linear{}'.format(idx): nn.Linear(self.hidden_dims[idx - 1], hidden_dim),
                        'activation{}'.format(idx): nn.Tanh()
                    }
                )
        layers.update(
            {
                'dropout{}'.format(len(self.hidden_dims)): nn.Dropout(p=self.dropout),
                'linear{}'.format(len(self.hidden_dims)): nn.Linear(
                    self.hidden_dims[len(self.hidden_dims) - 1], 1
                ),
                'activation{}'.format(len(self.hidden_dims)): nn.Sigmoid()
            }
        )

        self.mlp = nn.Sequential(layers)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        q_mask = kwargs["q_mask"]
        h_impact = torch.sigmoid(kwargs['other']['knowledge_impact'])
        g_impact = torch.sigmoid(self.g_impact_a * h_impact +
                                 self.g_impact_b * torch.sigmoid(diff_ts) * torch.sigmoid(disc_ts))
        input_x = torch.sigmoid(disc_ts) * (torch.sigmoid(student_ts) + g_impact - torch.sigmoid(diff_ts)) * q_mask
        return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        return torch.sigmoid(mastery)

    def monotonicity(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)
