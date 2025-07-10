import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from ..._base import _CognitiveDiagnosisModel
from ...interfunc import get_interaction_function
from ...extractor import ORCDF_EX


class ORCDF(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        Oversmoothing-Resistant Cognitive Diagnosis Framework (ORCDF)
        Shuo Liu et al. ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems. KDD'24.
        """
        super().__init__(config=config)
        self.extractor = ORCDF_EX(config)
        inter_func_class, require_transfer = get_interaction_function(config.get('if_type', 'KANCD_IF'))
        self.inter_func = inter_func_class(config)
        self.config['require_transfer'] = require_transfer
        self.build_graph()

    def build_graph(self):
        datahub = self.config['datahub']
        ek_graph = datahub.q_matrix.copy()

        se_graph_right, se_graph_wrong = [self.__create_adj_se(datahub['train'], is_subgraph=True)[i] for i in
                                          range(2)]
        se_graph = self.__create_adj_se(datahub['train'], is_subgraph=False)

        if self.config['flip_ratio']:
            def get_flip_data():
                np_response_flip = datahub['train'].copy()
                column = np_response_flip[:, 2]
                probability = np.random.choice([True, False], size=column.shape,
                                               p=[self.config['flip_ratio'], 1 - self.config['flip_ratio']])
                column[probability] = 1 - column[probability]
                np_response_flip[:, 2] = column
                return np_response_flip

        graph_dict = {
            'right': self.__final_graph(se_graph_right, ek_graph),
            'wrong': self.__final_graph(se_graph_wrong, ek_graph),
            'response': datahub['train'],
            'Q_Matrix': datahub.q_matrix.copy(),
            'flip_ratio': self.config['flip_ratio'],
            'all': self.__final_graph(se_graph, ek_graph)
        }

        self.extractor.get_graph_dict(graph_dict)


    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])


    def update_graph(self, data, q_matrix):
        se_graph_right, se_graph_wrong = [self.__create_adj_se(data, is_subgraph=True)[i] for i in
                                          range(2)]
        self.extractor.graph_dict['right'] = self.__final_graph(se_graph_right, q_matrix)
        self.extractor.graph_dict['wrong'] = self.__final_graph(se_graph_wrong, q_matrix)

    @staticmethod
    def __get_csr(rows, cols, shape):
        values = np.ones_like(rows)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def __sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo()
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce().float()

    def __create_adj_se(self, np_response, is_subgraph=False):
        if is_subgraph:
            train_stu_right = np_response[np_response[:, 2] == 1, 0]
            train_exer_right = np_response[np_response[:, 2] == 1, 1]
            train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
            train_exer_wrong = np_response[np_response[:, 2] == 0, 1]

            adj_se_right = self.__get_csr(train_stu_right, train_exer_right,
                                          shape=(self.config['student_num'], self.config['exercise_num']))
            adj_se_wrong = self.__get_csr(train_stu_wrong, train_exer_wrong,
                                          shape=(self.config['student_num'], self.config['exercise_num']))
            return adj_se_right.toarray(), adj_se_wrong.toarray()

        else:
            response_stu = np_response[:, 0]
            response_exer = np_response[:, 1]
            adj_se = self.__get_csr(response_stu, response_exer, shape=(self.config['student_num'], self.config['exercise_num']))
            return adj_se.toarray()

    def __final_graph(self, se, ek):
        sek_num = self.config['student_num'] + self.config['exercise_num'] + self.config['knowledge_num']
        se_num = self.config['student_num'] + self.config['exercise_num']
        tmp = np.zeros(shape=(sek_num, sek_num))
        tmp[:self.config['student_num'], self.config['student_num']: se_num] = se
        tmp[self.config['student_num']:se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self.__sp_mat_to_sp_tensor(adj_matrix)


