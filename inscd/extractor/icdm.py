import torch
import torch.nn as nn
import torch.nn.functional as F
from .._base import _Extractor
import dgl
from dgl.base import DGLError
from dgl import function as fn
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.nn.pytorch import GATConv, GATv2Conv


class SAGEConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            aggregator_type,
            feat_drop=0.0,
            bias=True,
            norm=None,
            activation=None,
    ):
        super(SAGEConv, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        with graph.local_scope():
            graph = graph.to(feat.device)
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][
                                             : graph.num_dst_nodes()
                                             ]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                        degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = h_neigh
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = graph.dstdata["neigh"]
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = h_self + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class Weighted_Summation(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Weighted_Summation, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim), requires_grad=True))
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class SAGENet(nn.Module):
    def __init__(self, dim, layers_num=2, type='mean', drop=True, d_1=0.05, d_2=0.1):
        super(SAGENet, self).__init__()
        self.drop = drop
        self.type = type
        self.d_1 = d_1
        self.d_2 = d_2
        self.layers = []
        for i in range(layers_num):
            if type == 'mean' or type == 'pool':
                self.layers.append(SAGEConv(in_feats=dim, out_feats=dim, aggregator_type=type))
            elif type == 'gat':
                self.layers.append(GATConv(in_feats=dim, out_feats=dim, num_heads=self.config['num_heads']))
            elif type == 'gatv2':
                self.layers.append(GATv2Conv(in_feats=dim, out_feats=dim, num_heads=self.config['num_heads']))

    def forward(self, g, h):
        outs = [h]
        tmp = h
        from dgl import DropEdge
        for index, layer in enumerate(self.layers):
            drop = DropEdge(p=self.d_1 + self.d_2 * index)
            if self.drop:
                if self.training:
                    g = drop(g)
                if self.type != 'mean' and self.type != 'pool':
                    g = dgl.add_self_loop(g)
                    tmp = torch.mean(layer(g, tmp), dim=1)
                else:
                    tmp = layer(g, tmp)
            else:
                if self.type != 'mean' and self.type != 'pool':
                    g = dgl.add_self_loop(g)
                    tmp = torch.mean(layer(g, tmp), dim=1)
                else:
                    tmp = layer(g, tmp)
            outs.append(tmp / (1 + index))
        res = torch.sum(torch.stack(
            outs, dim=1), dim=1)
        return res


class ICDM_EX(_Extractor, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.student_num = config['student_num']
        self.exercise_num = config['exercise_num']
        self.knowledge_num = config['knowledge_num']
        self.latent_dim = config['latent_dim']

        self.gcn_layers = config['gcn_layers']
        self.d_1 = config['d_1']
        self.d_2 = config['d_2']
        self.khop = config['khop']
        self.gcn_drop = True
        self.is_glif = config['is_glif']
        self.graph = ...
        self.Involve_Matrix = ...
        self.norm_adj = ...
        self.__student_emb = nn.Embedding(self.student_num, self.latent_dim)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, self.latent_dim)
        self.__exercise_right_emb = nn.Embedding(self.exercise_num, self.latent_dim)
        self.__exercise_wrong_emb = nn.Embedding(self.exercise_num, self.latent_dim)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1)
        self.__knowledge_impact_emb = nn.Embedding(self.exercise_num, self.latent_dim)
        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff_right": self.__exercise_right_emb.weight,
            "diff_wrong": self.__exercise_wrong_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight
        }
        self.drop = True

        self.S_E_right = SAGENet(dim=self.latent_dim, type=self.config['agg_type'], layers_num=self.khop, drop=self.drop,
                                 d_1=self.d_1, d_2=self.d_2)
        self.S_E_wrong = SAGENet(dim=self.latent_dim, type=self.config['agg_type'], layers_num=self.khop, drop=self.drop,
                                 d_1=self.d_1, d_2=self.d_2)
        self.E_C_right = SAGENet(dim=self.latent_dim, type=self.config['agg_type'], layers_num=self.khop, drop=self.drop,
                                 d_1=self.d_1, d_2=self.d_2)
        self.E_C_wrong = SAGENet(dim=self.latent_dim, type=self.config['agg_type'], layers_num=self.khop, drop=self.drop,
                                 d_1=self.d_1, d_2=self.d_2)
        self.S_C = SAGENet(dim=self.latent_dim, type=self.config['agg_type'], layers_num=self.khop, drop=self.drop,
                           d_1=self.d_1,
                           d_2=self.d_2)

        self.attn_S = Weighted_Summation(self.latent_dim, attn_drop=0.2)
        self.attn_E_right = Weighted_Summation(self.latent_dim, attn_drop=0.2)
        self.attn_E_wrong = Weighted_Summation(self.latent_dim, attn_drop=0.2)
        self.attn_E = Weighted_Summation(self.latent_dim, attn_drop=0.2)
        self.attn_C = Weighted_Summation(self.latent_dim, attn_drop=0.2)

        self.transfer_student_layer = nn.Linear(self.latent_dim, self.knowledge_num)
        self.transfer_exercise_layer = nn.Linear(self.latent_dim, self.knowledge_num)
        self.transfer_knowledge_layer = nn.Linear(self.latent_dim, self.knowledge_num)

        self.apply(self.initialize_weights)

    def get_graph_dict(self, graph):
        self.graph = graph
        self.Involve_Matrix = self.dgl2tensor(self.graph['I'])[:self.student_num, self.student_num:]

    def get_norm_adj(self, norm_adj):
        self.norm_adj = norm_adj

    @staticmethod
    def dgl2tensor(g):
        import networkx as nx
        nx_graph = g.to_networkx()
        adj_matrix = nx.to_numpy_array(nx_graph)
        tensor = torch.from_numpy(adj_matrix)
        return tensor

    @staticmethod
    def get_subgraph(g, id):
        return dgl.in_subgraph(g, id)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def __common_forward(self, student_id, exercise_id, q_mask=None, concept_id=None):
        if q_mask is None:
            concept_id = concept_id
        else:
            concept_id = torch.where(q_mask != 0)[1]
        concept_id_S = concept_id + torch.full(concept_id.shape, self.student_num).to(concept_id.device)
        concept_id_E = concept_id + torch.full(concept_id.shape, self.exercise_num).to(concept_id.device)
        exercise_id_S = exercise_id + torch.full(exercise_id.shape, self.student_num).to(concept_id.device)

        subgraph_node_id_Q = torch.cat((exercise_id.detach().cpu(), concept_id_E.detach().cpu()), dim=-1)
        subgraph_node_id_R = torch.cat((student_id.detach().cpu(), exercise_id_S.detach().cpu()), dim=-1)
        subgraph_node_id_I = torch.cat((student_id.detach().cpu(), concept_id_S.detach().cpu()), dim=-1)

        R_subgraph_Right = self.get_subgraph(self.graph['right'], subgraph_node_id_R)
        R_subgraph_Wrong = self.get_subgraph(self.graph['wrong'], subgraph_node_id_R)
        I_subgraph = self.get_subgraph(self.graph['I'], subgraph_node_id_I)
        Q_subgraph = self.get_subgraph(self.graph['Q'], subgraph_node_id_Q)

        exer_info_right = self.__exercise_right_emb.weight
        exer_info_wrong = self.__exercise_wrong_emb.weight
        concept_info = self.__knowledge_emb.weight

        E_C_right = torch.cat([exer_info_right, concept_info])
        E_C_wrong = torch.cat([exer_info_wrong, concept_info])

        E_C_info_right = self.E_C_right(Q_subgraph, E_C_right)
        E_C_info_wrong = self.E_C_wrong(Q_subgraph, E_C_wrong)
        #
        stu_info = self.__student_emb.weight
        S_C = torch.cat([stu_info, concept_info])
        S_E_right = torch.cat([stu_info, exer_info_right])
        S_E_wrong = torch.cat([stu_info, exer_info_wrong])
        S_E_info_right = self.S_E_right(R_subgraph_Right, S_E_right)
        S_E_info_wrong = self.S_E_wrong(R_subgraph_Wrong, S_E_wrong)
        S_C_info = self.S_C(I_subgraph, S_C)

        E_forward_right = self.attn_E_right.forward(
            [E_C_info_right[:self.exercise_num], S_E_info_right[self.student_num:]])
        E_forward_wrong = self.attn_E_wrong.forward(
            [E_C_info_wrong[:self.exercise_num], S_E_info_wrong[self.student_num:]])
        E_forward = E_forward_right * E_forward_wrong
        C_forward = self.attn_C.forward(
            [E_C_info_right[self.exercise_num:], E_C_info_wrong[self.exercise_num:], S_C_info[self.student_num:]])
        S_forward = self.attn_S.forward(
            [S_E_info_right[:self.student_num], S_E_info_wrong[:self.student_num], S_C_info[:self.student_num]])
        if self.is_glif:
            emb = torch.cat([S_forward, E_forward])
            out = self.conv(emb)
            S_forward, E_forward = out[:self.student_num], out[self.student_num:]
        return S_forward, E_forward, C_forward

    def extract(self, student_id, exercise_id, q_mask):
        S_forward, E_forward, C_forward = self.__common_forward(student_id, exercise_id, q_mask, None)
        student_ts, diff_ts, knowledge_ts = self.transfer_student_layer(
            S_forward)[student_id], self.transfer_exercise_layer(E_forward)[exercise_id], self.transfer_knowledge_layer(C_forward)
        disc_ts = self.__disc_emb(exercise_id)
        knowledge_impact_ts = self.__knowledge_impact_emb(exercise_id)
        return student_ts, diff_ts, disc_ts, knowledge_ts, {'extra_loss': 0,
                                                            'knowledge_impact': knowledge_impact_ts}

    def conv(self, emb):
        all_emb = emb
        embs = [emb]
        for layer in range(self.gcn_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
        out_embs = torch.mean(torch.stack(embs, dim=1), dim=1)
        return out_embs

    @staticmethod
    def concept_distill(matrix, concept):
        coeff = 1.0 / torch.sum(matrix, dim=1)
        concept = matrix.to(torch.float64) @ concept
        concept_distill = concept * coeff[:, None]
        return concept_distill

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        student_id = torch.arange(self.student_num)
        exercise_id = torch.arange(self.exercise_num)
        concept_id = torch.arange(self.knowledge_num)
        S_forward, E_forward, C_forward = self.__common_forward(student_id, exercise_id, None, concept_id)

        student_ts = self.transfer_student_layer(S_forward)
        diff_ts = self.transfer_exercise_layer(E_forward)
        knowledge_ts = self.transfer_knowledge_layer(C_forward)

        disc_ts = self.__disc_emb.weight
        self.__emb_map["mastery"] = student_ts
        self.__emb_map["diff"] = diff_ts
        self.__emb_map["disc"] = disc_ts
        self.__emb_map["knowledge"] = knowledge_ts
        return self.__emb_map[item]
