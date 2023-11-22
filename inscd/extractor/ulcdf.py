import torch
import torch.nn as nn
import torch.nn.functional as F
from .._base import _Extractor


class ULCDF_EXTRACTOR(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, latent_dim: int, device,
                 dtype, gcn_layers=3, keep_prob=0.9, leaky=0.8, mode='all'):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim

        self.device = device
        self.dtype = dtype
        self.gcn_layers = gcn_layers
        self.keep_prob = keep_prob
        self.leaky = leaky
        self.mode = mode
        self.gcn_drop = True
        torch.set_default_dtype(torch.float64)

        self.__student_emb = nn.Embedding(self.student_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__exercise_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)
        self.__emb_map = {}
        # self.__emb_map = {
        #     "student": self.__student_emb,
        #     "exercise": self.__exercise_emb,
        #     "disc": self.__disc_emb,
        #     "knowledge": self.__knowledge_emb
        # }

        self.concat_layer = nn.Linear(2 * self.latent_dim, self.latent_dim).to(self.device)
        self.concat_layer_1 = nn.Linear(2 * self.latent_dim, self.latent_dim).to(self.device)
        self.transfer_student_layer = nn.Linear(self.latent_dim, self.knowledge_num).to(self.device)
        self.transfer_exercise_layer = nn.Linear(self.latent_dim, self.knowledge_num).to(self.device)
        self.transfer_knowledge_layer = nn.Linear(self.latent_dim, self.knowledge_num).to(self.device)
        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_graph_dict(self, graph_dict):
        self.graph_dict = graph_dict

    def convolution(self, graph):
        stu_emb, exer_emb, know_emb = (self.__student_emb.weight, self.__exercise_emb.weight
                                       , self.__knowledge_emb.weight)
        all_emb = torch.cat([stu_emb, exer_emb, know_emb]).to(self.device)
        embs = [all_emb]
        for layer in range(self.gcn_layers):
            all_emb = torch.sparse.mm(self.graph_drop(graph), all_emb)
            embs.append(all_emb)
        out_embs = torch.mean(torch.stack(embs, dim=1), dim=1)
        return out_embs

    def _common_foward(self):
        out_hol_embs, right_embs, wrong_embs = self.convolution(self.graph_dict['all']), self.convolution(
            self.graph_dict['right']), self.convolution(self.graph_dict['wrong'])
        if self.mode == 'dis':
            out_embs = F.leaky_relu(self.concat_layer(torch.cat([right_embs, wrong_embs], dim=1)),
                                    negative_slope=self.leaky)
        elif self.mode == 'hol':
            out_embs = out_hol_embs
        else:
            out_dis_embs = F.leaky_relu(self.concat_layer(torch.cat([right_embs, wrong_embs], dim=1)),
                                        negative_slope=self.leaky)
            out_embs = F.leaky_relu(
                self.concat_layer_1(torch.cat([out_dis_embs, out_hol_embs], dim=1)),
                negative_slope=self.leaky)

        stus, exers, knows = torch.split(out_embs, [self.student_num, self.exercise_num, self.knowledge_num])
        return stus, exers, knows

    def _dropout(self, graph, keep_prob):
        if self.gcn_drop and self.training:
            size = graph.size()
            index = graph.indices().t()
            values = graph.values()
            random_index = torch.rand(len(values)) + keep_prob
            random_index = random_index.int().bool()
            index = index[random_index]
            values = values[random_index] / keep_prob
            g = torch.sparse.DoubleTensor(index.t(), values, size)
            return g
        else:
            return graph

    def graph_drop(self, graph):
        g_droped = self._dropout(graph, self.keep_prob)
        return g_droped

    def extract(self, student_id, exercise_id, q_mask):
        stus, exers, knows = self._common_foward()
        batch_stus_forward = self.transfer_student_layer(
            F.embedding(student_id, stus)) if self.mode != 'tf' else F.embedding(
            student_id, stus)
        batch_exers_forward = self.transfer_exercise_layer(
            F.embedding(exercise_id, exers)) if self.mode != 'tf' else F.embedding(exercise_id, exers)
        knows_forward = self.transfer_knowledge_layer(knows) if self.mode != 'tf' else knows
        disc_ts = self.__disc_emb(exercise_id)

        return batch_stus_forward, batch_exers_forward, disc_ts, knows_forward

    def set_results(self):
        stus, exers, knows = self._common_foward()
        stus_forward = self.transfer_student_layer(
            stus) if self.mode != 'tf' else stus
        exers_forward = self.transfer_exercise_layer(
           exers) if self.mode != 'tf' else  exers
        knows_forward = self.transfer_knowledge_layer(knows) if self.mode != 'tf' else knows
        self.__emb_map['mastery'] = stus_forward
        self.__emb_map['diff'] = exers_forward
        self.__emb_map['knowledge'] = knows_forward

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can detach {} from embeddings.".format(self.__emb_map.keys()))
        return self.__emb_map[item]
        # return torch.sigmoid(self.__emb_map[item].weight.detach().cpu()).numpy()
