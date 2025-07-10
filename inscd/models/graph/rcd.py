# import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from ..._base import _CognitiveDiagnosisModel
from ...interfunc import get_interaction_function
from ...extractor import RCD_EX


class RCD(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        Relation Map-driven Cognitive Diagnosis Model (RCD)
        Weibo Gao et al. RCD: Relation Map Driven Cognitive Diagnosis for Intelligent Education Systems. SIGIR'21
        """
        super().__init__(config=config)
        self.extractor = RCD_EX(config)
        inter_func_class, require_transfer = get_interaction_function(config.get('if_type', 'RCD_IF'))
        self.inter_func = inter_func_class(config)
        self.config['require_transfer'] = require_transfer
        local_map = {
            'k_from_e': self.build_graph4ke(config['datahub'], from_e=True),
            'e_from_k': self.build_graph4ke(config['datahub'], from_e=False),
            'e_from_s': self.build_graph4se(config['datahub'], from_s=True),
            's_from_e': self.build_graph4se(config['datahub'], from_s=False),
        }
        self.extractor.get_local_map(local_map)



    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])


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
        node = self.config['self.knowledge_num']
        g.add_nodes(node)
        edge_list = []
        src_idx_np, tar_idx_np = np.where(datahub['directed_graph'] != 0)
        for src_indx, tar_index in zip(src_idx_np.tolist(), tar_idx_np.tolist()):
            edge_list.append((int(src_indx), int(tar_index)))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g