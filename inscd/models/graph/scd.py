import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from ..._base import _CognitiveDiagnosisModel
from ...interfunc import get_interaction_function
from ...extractor import SCD_EX


class SCD(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        Self-supervised Cognitive Diagnosis Model (SCD)
        Shanshan Wang et al. Self-Supervised Graph Learning for Long-Tailed Cognitive Diagnosis. AAAI'23.
        """
        super().__init__(config=config)
        self.extractor = SCD_EX(config)
        inter_func_class, require_transfer = get_interaction_function(config.get('if_type', 'SCD_IF'))
        self.inter_func = inter_func_class(config)
        self.config['require_transfer'] = require_transfer
        self.build_graph()

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

    def build_graph(self):
        graph = {
            'k_from_e': self.build_graph4ke(self.config['datahub'], from_e=True),
            'e_from_k': self.build_graph4ke(self.config['datahub'], from_e=False),
            'e_from_s': self.build_graph4se(self.config['datahub'], from_s=True),
            's_from_e': self.build_graph4se(self.config['datahub'], from_s=False),
        }
        graph_1 = {
            'k_from_e': self.build_graph4ke(self.config['datahub'], from_e=True),
            'e_from_k': self.build_graph4ke(self.config['datahub'], from_e=False),
            'e_from_s': self.drop_edges_based_on_degree(self.build_graph4se(self.config['datahub'], from_s=True)),
            's_from_e': self.drop_edges_based_on_degree(self.build_graph4se(self.config['datahub'], from_s=False)),
        }
        graph_2 = {
            'k_from_e': self.build_graph4ke(self.config['datahub'], from_e=True),
            'e_from_k': self.build_graph4ke(self.config['datahub'], from_e=False),
            'e_from_s': self.drop_edges_based_on_degree(self.build_graph4se(self.config['datahub'], from_s=True)),
            's_from_e': self.drop_edges_based_on_degree(self.build_graph4se(self.config['datahub'], from_s=False)),
        }
        graph_list = [graph, graph_1, graph_2]
        self.extractor.get_graph_list(graph_list)

    @staticmethod
    def drop_edges_based_on_degree(graph, pmin=0.4, k=2):
        degrees = graph.in_degrees()
        edge_mask = torch.ones(graph.number_of_edges(), dtype=torch.bool)
        def calculate_importance(data):
            data = k / torch.log(data + 1 + 10e-5)
            return torch.clamp(data, min=pmin)
        drop_p = calculate_importance(degrees)
        for idx, p in enumerate(drop_p):
            if p < pmin:
                drop_p[idx] = pmin

        for edge_id in range(graph.number_of_edges()):
            src, dst = graph.find_edges(edge_id)
            drop_rate_dst = drop_p[dst]
            if torch.rand(1) < drop_rate_dst:
                edge_mask[edge_id] = False

        src, dst = graph.edges()
        src = src[edge_mask]
        dst = dst[edge_mask]
        new_graph = dgl.graph((src, dst), num_nodes=graph.number_of_nodes())
        return new_graph

    def build_graph4ke(self, datahub, from_e: bool):
        q = datahub.q_matrix.copy()
        node = self.config['knowledge_num'] + self.config['exercise_num']
        g = dgl.DGLGraph()
        g.add_nodes(node)
        edge_list = []
        indices = np.where(q != 0)
        if from_e:
            for exer_id, know_id in zip(indices[0].tolist(), indices[1].tolist()):
                edge_list.append((int(exer_id), int(know_id + self.config['exercise_num'] - 1)))
        else:
            for exer_id, know_id in zip(indices[0].tolist(), indices[1].tolist()):
                edge_list.append((int(know_id + self.config['exercise_num'] - 1), int(exer_id)))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g

    def build_graph4se(self, datahub, from_s: bool):
        np_train = datahub['train']
        node = self.config['student_num'] + self.config['exercise_num']
        g = dgl.DGLGraph()
        g.add_nodes(node)
        edge_list = []
        for index in range(np_train.shape[0]):
            stu_id = np_train[index, 0]
            exer_id = np_train[index, 1]
            if from_s:
                edge_list.append((int(stu_id + self.config['exercise_num'] - 1), int(exer_id)))
            else:
                edge_list.append((int(exer_id), int(stu_id + self.config['exercise_num'] - 1)))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g

    def build_graph4di(self, datahub):
        g = dgl.DGLGraph()
        node = self.config['knowledge_num']
        g.add_nodes(node)
        edge_list = []
        src_idx_np, tar_idx_np = np.where(datahub['directed_graph'] != 0)
        for src_indx, tar_index in zip(src_idx_np.tolist(), tar_idx_np.tolist()):
            edge_list.append((int(src_indx), int(tar_index)))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
