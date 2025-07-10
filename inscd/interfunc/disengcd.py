import torch
import torch.nn as nn

from collections import OrderedDict

from ._util import none_neg_clipper
from .._base import _InteractionFunction

class DISENGCD_IF(_InteractionFunction, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.knowledge_num = config['knowledge_num']
        self.prednet_full1 = nn.Linear(self.knowledge_num, 1)
        self.prednet_full2 = nn.Linear(self.knowledge_num, 1)
        self.prednet_full3 = nn.Linear(self.knowledge_num,self.knowledge_num)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        knowledge_ts = kwargs['knowledge_ts']
        q_mask = kwargs["q_mask"]
        batch_exer_vector = diff_ts.repeat(1, diff_ts.shape[1]).reshape(diff_ts.shape[0],
                                                                                      diff_ts.shape[1],
                                                                                      diff_ts.shape[1])
        batch_stu_vector = student_ts.repeat(1, student_ts.shape[1]).reshape(student_ts.shape[0],
                                                                                   student_ts.shape[1],
                                                                                   student_ts.shape[1])
        kn_vector = knowledge_ts.repeat(student_ts.shape[0], 1).reshape(student_ts.shape[0], knowledge_ts.shape[0],
                                                                     knowledge_ts.shape[1])

        alpha = self.prednet_full1(batch_stu_vector + kn_vector).squeeze(2) 
        betta = self.prednet_full2(batch_exer_vector + kn_vector).squeeze(2) 
        o = torch.sigmoid(self.prednet_full3(alpha * betta))

        sum_out = torch.sum(o * q_mask, dim=1)             
        count_of_concept = torch.sum(q_mask, dim=1)        
        return sum_out/count_of_concept

    def transform(self, mastery, knowledge):
        self.eval()
        blocks = torch.split(torch.arange(mastery.shape[0]), 5)
        mas = []
        for block in blocks:
            batch, dim = mastery[block].size()
            stu_emb = mastery[block].view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
            knowledge_emb = knowledge.repeat(batch, 1).view(batch, self.knowledge_num, -1)
            mas.append(torch.sigmoid(self.prednet_full1(stu_emb+knowledge_emb)).view(batch, -1))
        return torch.vstack(mas)

    def monotonicity(self):
        self.prednet_full1.apply(none_neg_clipper)
        self.prednet_full2.apply(none_neg_clipper)
        self.prednet_full3.apply(none_neg_clipper)
