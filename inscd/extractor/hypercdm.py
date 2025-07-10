import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .._base import _Extractor

import scipy.sparse as sp


class HyperCDM_EX(_Extractor, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.student_num = config['student_num']
        self.exercise_num = config['exercise_num']
        self.knowledge_num = config['knowledge_num']
        emb_dim = config['emb_dim']
        self.latent_dim = emb_dim
        feature_dim = config['feat_dim']
        self.layers = config['layers']

        self.__student_emb = nn.Embedding(self.student_num, self.latent_dim)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, self.latent_dim)
        self.__exercise_emb = nn.Embedding(self.exercise_num, self.latent_dim)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1)
        self.__knowledge_impact_emb = nn.Embedding(self.exercise_num, self.latent_dim)

        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff": self.__exercise_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight
        }
        self.student_emb2feature = nn.Linear(emb_dim, feature_dim)
        self.exercise_emb2feature = nn.Linear(emb_dim, feature_dim)
        self.knowledge_emb2feature = nn.Linear(emb_dim, feature_dim)
        self.exercise_emb2discrimination = nn.Linear(emb_dim, 1)

        self.apply(self.initialize_weights)

    def get_graph_dict(self, graph_dict):
        self.graph_dict = graph_dict

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)


    def convolution(self, embedding, adj):
        all_emb = embedding.weight
        final = [all_emb]
        for i in range(self.layers):
            # implement momentum hypergraph convolution
            all_emb = torch.sparse.mm(adj, all_emb) + self.config['momentum']* all_emb
            final.append(all_emb)
        final_emb = torch.mean(torch.stack(final, dim=1), dim=1)
        return final_emb

    def __common_forward(self):
        convolved_student_emb = self.convolution(self.__student_emb, self.graph_dict['student_adj'].to(self.__emb_map['mastery'].device))
        convolved_exercise_emb = self.convolution(self.__exercise_emb, self.graph_dict['exercise_adj'].to(self.__emb_map['mastery'].device))
        convolved_knowledge_emb = self.convolution(self.__knowledge_emb, self.graph_dict['knowledge_adj'].to(self.__emb_map['mastery'].device))
        return convolved_student_emb, convolved_exercise_emb, convolved_knowledge_emb


    def extract(self, student_id, exercise_id, q_mask):
        stu_forward, exer_forward, know_forward = self.__common_forward()
        batch_student = F.embedding(student_id, stu_forward)
        batch_exercise = F.embedding(exercise_id, exer_forward)

        student_ts= F.leaky_relu(self.student_emb2feature(batch_student), negative_slope=self.config['leak'])
        diff_ts = F.leaky_relu(self.exercise_emb2feature(batch_exercise), negative_slope=self.config['leak'])
        knowledge_ts = F.leaky_relu(self.knowledge_emb2feature(know_forward), negative_slope=self.config['leak'])
        disc_ts = torch.sigmoid(self.exercise_emb2discrimination(batch_exercise))
        student_ts = student_ts @ knowledge_ts.T
        diff_ts = diff_ts @ knowledge_ts.T

        knowledge_impact_ts = self.__knowledge_impact_emb(exercise_id)
        return student_ts, diff_ts, disc_ts, knowledge_ts, {'extra_loss': 0,
                                                            'knowledge_impact': knowledge_impact_ts}
    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        stu_forward, exer_forward, know_forward = self.__common_forward()


        student_ts= F.leaky_relu(self.student_emb2feature(stu_forward), negative_slope=self.config['leak'])
        diff_ts = F.leaky_relu(self.exercise_emb2feature(exer_forward), negative_slope=self.config['leak'])
        knowledge_ts = F.leaky_relu(self.knowledge_emb2feature(know_forward), negative_slope=self.config['leak'])
        student_ts = student_ts @ knowledge_ts.T
        diff_ts = diff_ts @ knowledge_ts.T
        disc_ts = torch.sigmoid(self.exercise_emb2discrimination(exer_forward))

        self.__emb_map["mastery"] = student_ts
        self.__emb_map["diff"] = diff_ts
        self.__emb_map["disc"] = disc_ts
        self.__emb_map["knowledge"] = knowledge_ts
        return self.__emb_map[item]
