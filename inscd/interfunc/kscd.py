import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import _InteractionFunction


class KSCD_IF(_InteractionFunction, nn.Module):
    def __init__(self, knowledge_num: int, latent_dim: int, device, dtype):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype
        self.disc_mlp = nn.Linear(self.latent_dim, 1, dtype=self.dtype).to(self.device)
        self.f_sk = nn.Linear(self.knowledge_num + self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.f_ek = nn.Linear(self.knowledge_num + self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.f_se = nn.Linear(self.knowledge_num, 1, dtype=self.dtype).to(self.device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        q_mask = kwargs["q_mask"]
        knowledge_ts = kwargs['knowledge_ts']
        stu_ability = torch.sigmoid(student_ts @ knowledge_ts.T)
        diff_emb = torch.sigmoid(diff_ts @ knowledge_ts.T)
        disc_ts = torch.sigmoid(self.disc_mlp(diff_ts))
        batch, dim = student_ts.size()
        stu_emb = stu_ability.unsqueeze(1).repeat(1, self.knowledge_num, 1)
        diff_emb = diff_emb.unsqueeze(1).repeat(1, self.knowledge_num, 1)
        Q_relevant = q_mask.unsqueeze(2).repeat(1, 1, self.knowledge_num)
        knowledge_emb = knowledge_ts.repeat(batch, 1).view(batch, self.knowledge_num, -1)
        s_k_concat = torch.sigmoid(self.f_sk(torch.cat([stu_emb, knowledge_emb], dim=-1)))
        e_k_concat = torch.sigmoid(self.f_ek(torch.cat([diff_emb, knowledge_emb], dim=-1)))
        return torch.sigmoid(disc_ts * self.f_se(torch.mean((s_k_concat - e_k_concat) * Q_relevant, dim=1))).view(-1)

    def transform(self, mastery, knowledge):
        return F.sigmoid(mastery @ knowledge.T)
