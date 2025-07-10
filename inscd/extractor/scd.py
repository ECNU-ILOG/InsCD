import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import _Extractor


class GraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': a}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        g = g.to(h.device)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class SCD_GraphLayer(nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int):
        self.knowledge_num = knowledge_num
        self.student_num = student_num
        self.exercise_num = exercise_num
        super(SCD_GraphLayer, self).__init__()
        self.e_from_s_gat = GraphLayer(self.knowledge_num, self.knowledge_num)
        self.s_from_e_gat = GraphLayer(self.knowledge_num, self.knowledge_num)
        self.e_from_k_gat = GraphLayer(self.knowledge_num, self.knowledge_num)
        self.k_from_e_gat = GraphLayer(self.knowledge_num, self.knowledge_num)
        self.e_attn0 = nn.Linear(2 * self.knowledge_num, 1, bias=True)
        self.e_attn1 = nn.Linear(2 * self.knowledge_num, 1, bias=True)
        self.s_attn0 = nn.Linear(2 * self.knowledge_num, 1, bias=True)
        self.k_attn0 = nn.Linear(2 * self.knowledge_num, 1, bias=True)

    def forward(self, stu_emb, exer_emb, kn_emb, graph_dict):
        e_k_weight = torch.cat((exer_emb, kn_emb), dim=0)
        s_e_weight = torch.cat((exer_emb, stu_emb), dim=0)

        s_from_e_conv = self.s_from_e_gat(graph_dict['s_from_e'], s_e_weight)
        e_from_s_conv = self.e_from_s_gat(graph_dict['e_from_s'], s_e_weight)
        e_from_k_conv = self.e_from_k_gat(graph_dict['e_from_k'], e_k_weight)
        k_from_e_conv = self.k_from_e_gat(graph_dict['k_from_e'], e_k_weight)

        score0 = self.s_attn0(torch.cat([stu_emb, s_from_e_conv[:self.student_num, :]], dim=1))
        score = F.softmax(score0, dim=1)
        ult_stu_emb = stu_emb + score[:, 0].unsqueeze(1) * s_from_e_conv[:self.student_num, :]

        score0 = self.e_attn0(torch.cat([exer_emb, e_from_s_conv[self.student_num:, :]], dim=1))
        score1 = self.e_attn1(torch.cat([exer_emb, e_from_k_conv[:self.exercise_num, :]], dim=1))
        score = F.softmax(torch.cat([score0, score1], dim=1), dim=1)
        ult_exer_emb = exer_emb + score[:, 0].unsqueeze(1) * e_from_s_conv[self.student_num:, :] + score[:,
                                                                                                   1].unsqueeze(
            1) * e_from_k_conv[:self.exercise_num, :]

        score0 = self.k_attn0(torch.cat([kn_emb, k_from_e_conv[self.exercise_num:, :]], dim=1))
        score = F.softmax(score0, dim=1)
        ult_kn_emb = kn_emb + score[:, 0].unsqueeze(1) * k_from_e_conv[self.exercise_num:, :]

        return ult_stu_emb, ult_exer_emb, ult_kn_emb


class SCD_EX(_Extractor, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.student_num = config['student_num']
        self.exercise_num = config['exercise_num']
        self.knowledge_num = config['knowledge_num']
        self.gcn_layers = config['gcn_layers']
        self.latent_dim = config.get('latent_dim', self.knowledge_num)
        self.alphas = config['alphas']
        self.alphae = config['alphae']

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
        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_graph_list(self, graph_list):

        self.graph_list = graph_list

        self.gnet1 = SCD_GraphLayer(self.student_num, self.exercise_num, self.knowledge_num,
                                    )
        self.gnet2 = SCD_GraphLayer(self.student_num, self.exercise_num, self.knowledge_num,
                                    )

    def __common_forward(self, graph_dict):
        stu_emb = self.__student_emb.weight
        exer_emb = self.__exercise_emb.weight
        kn_emb = self.__knowledge_emb.weight
        stu_emb1, exer_emb1, kn_emb1 = self.gnet1(stu_emb, exer_emb, kn_emb, graph_dict)
        ult_stu_emb, ult_exer_emb, ult_kn_emb = self.gnet2(stu_emb1, exer_emb1, kn_emb1, graph_dict)
        return ult_stu_emb, ult_exer_emb, ult_kn_emb

    def contrastive_loss(self, h1, h2, mode='dp'):
        t = 0.5
        batch_size = h1.shape[0]
        negatives_mask = (~torch.eye(batch_size, batch_size,
                                     dtype=bool)).float().to(h1.device)
        z1 = F.normalize(h1, dim=1)
        z2 = F.normalize(h2, dim=1)
        if mode == 'cosine':
            similarity_matrix1 = F.cosine_similarity(
            z1.unsqueeze(1), z2.unsqueeze(0), dim=2)
        else:
            similarity_matrix1 = z1 @ z2.T
        positives = torch.exp(torch.diag(similarity_matrix1) / t)
        negatives = negatives_mask * torch.exp(similarity_matrix1 / t)
        loss_partial = -torch.log(positives / (positives + torch.sum(negatives, dim=1)))
        loss = torch.sum(loss_partial) / batch_size
        return loss

    def extract(self, student_id, exercise_id, q_mask):
        stu_forward, exer_forward, knows_forward = self.__common_forward(graph_dict=self.graph_list[0])
        if self.training:
            stu_emb_1, exer_emb_1, kn_emb_1 = self.__common_forward(graph_dict=self.graph_list[1])
            stu_emb_2, exer_emb_2, kn_emb_2 = self.__common_forward(graph_dict=self.graph_list[2])

            c_s_h1_loss = self.contrastive_loss(stu_emb_1, stu_emb_2)
            c_s_h2_loss = self.contrastive_loss(stu_emb_2, stu_emb_1)
            c_e_h1_loss = self.contrastive_loss(exer_emb_1, exer_emb_2)
            c_e_h2_loss = self.contrastive_loss(exer_emb_2, exer_emb_1)
            extra_loss = (self.alphas * (c_s_h1_loss + c_s_h2_loss) +
                          self.alphae * (c_e_h1_loss + c_e_h2_loss))
        else:
            extra_loss = 0

        batch_stu_emb = stu_forward[student_id]
        batch_exer_emb = exer_forward[exercise_id]
        disc_ts = self.__disc_emb(exercise_id)

        batch_stu_ts = batch_stu_emb
        batch_exer_ts = batch_exer_emb
        knowledge_ts = knows_forward

        return batch_stu_ts, batch_exer_ts, disc_ts, knowledge_ts, {'extra_loss': extra_loss}

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        stu_forward, exer_forward, knows_forward = self.__common_forward(graph_dict=self.graph_list[0])
        student_ts = stu_forward.repeat(1, stu_forward.shape[1]).reshape(stu_forward.shape[0],
                                                                         stu_forward.shape[1],
                                                                         stu_forward.shape[1])

        # get batch exercise data
        diff_ts = exer_forward.repeat(1, exer_forward.shape[1]).reshape(exer_forward.shape[0],
                                                                        exer_forward.shape[1],
                                                                        exer_forward.shape[1])

        # get batch knowledge concept data
        knowledge_ts = knows_forward.repeat(stu_forward.shape[0], 1).reshape(stu_forward.shape[0],
                                                                             knows_forward.shape[0],
                                                                             knows_forward.shape[1])

        disc_ts = self.__disc_emb.weight
        self.__emb_map["mastery"] = student_ts
        self.__emb_map["diff"] = diff_ts
        self.__emb_map["disc"] = disc_ts
        self.__emb_map["knowledge"] = knowledge_ts
        return self.__emb_map[item]
