# import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from ..._base import _CognitiveDiagnosisModel
from ...interfunc import get_interaction_function
from ...extractor import ICDM_EX


class ICDM(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        ICDM ...

        Parameters:
        student_num: int type
            The number of students in the response logs
        exercise_num: int type
            The number of exercises in the response logs
        knowledge_num: int type
            The number of knowledge concepts in the response logs
        method: Ignored
            Not used, present here for API consistency by convention.
        """
        super().__init__(config=config)
        self.extractor = ICDM_EX(config)
        inter_func_class, require_transfer = get_interaction_function(config.get('if_type', 'NCD_IF'))
        self.inter_func = inter_func_class(config)
        self.config['require_transfer'] = require_transfer

        right, wrong = self.build_graph4SE()
        graph = {
            'right': right,
            'wrong': wrong,
            'Q': self.build_graph4CE(),
            'I': self.build_graph4SC()
        }
        self.extractor.get_graph_dict(graph)
        self.extractor.get_norm_adj(self.create_adj_mat())
   
    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

    def build_graph4CE(self):
        node = self.config['exercise_num'] + self.config['knowledge_num']
        g = dgl.DGLGraph()
        g.add_nodes(node)
        edge_list = []
        indices = np.where(self.config['datahub'].q_matrix != 0)
        for exer_id, know_id in zip(indices[0].tolist(), indices[1].tolist()):
            edge_list.append((int(know_id + self.config['exercise_num']), int(exer_id)))
            edge_list.append((int(exer_id), int(know_id + self.config['exercise_num'])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g

    def build_graph4SE(self):
        node = self.config['student_num'] + self.config['exercise_num']
        g_right, g_wrong = dgl.DGLGraph(), dgl.DGLGraph()
        g_right.add_nodes(node)
        g_wrong.add_nodes(node)
        right_edge_list, wrong_edge_list = [], []
        data = self.config['datahub']['train']
        for index in range(data.shape[0]):
            stu_id = data[index, 0]
            exer_id = data[index, 1]
            if int(data[index, 2]) == 1:
                right_edge_list.append((int(stu_id), int(exer_id + self.config['student_num'])))
                right_edge_list.append((int(exer_id + self.config['student_num']), int(stu_id)))
            else:
                wrong_edge_list.append((int(stu_id), int(exer_id + self.config['student_num'])))
                wrong_edge_list.append((int(exer_id + self.config['student_num']), int(stu_id)))
        right_src, right_dst = tuple(zip(*right_edge_list))
        wrong_src, wrong_dst = tuple(zip(*wrong_edge_list))
        g_right.add_edges(right_src, right_dst)
        g_wrong.add_edges(wrong_src, wrong_dst)
        return g_right, g_wrong

    def build_graph4SC(self):
        node = self.config['student_num'] + self.config['knowledge_num']
        g = dgl.DGLGraph()
        g.add_nodes(node)
        edge_list = []
        sc_matrix = np.zeros(shape=(self.config['student_num'], self.config['knowledge_num']))
        data = self.config['datahub']['train']
        for index in range(data.shape[0]):
            stu_id = data[index, 0]
            exer_id = data[index, 1]
            concepts = np.where(self.config['datahub'].q_matrix[int(exer_id)] != 0)[0]
            for concept_id in concepts:
                if sc_matrix[int(stu_id), int(concept_id)] != 1:
                    edge_list.append((int(stu_id), int(concept_id + self.config['student_num'])))
                    edge_list.append((int(concept_id + self.config['student_num']), int(stu_id)))
                    sc_matrix[int(stu_id), int(concept_id)] = 1
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g

    @staticmethod
    def get_adj_matrix(tmp_adj):
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix


    @staticmethod
    def sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()


    def create_adj_mat(self):
        n_nodes = self.config['student_num'] + self.config['exercise_num']
        np_train = self.config['datahub']['train']
        train_stu = np_train[:, 0]
        train_exer = np_train[:, 1]
        ratings = np.ones_like(train_stu, dtype=np.float64)
        tmp_adj = sp.csr_matrix((ratings, (train_stu, train_exer + self.config['student_num'])), shape=(n_nodes, n_nodes))
        return self.sp_mat_to_sp_tensor(self.get_adj_matrix(tmp_adj))
