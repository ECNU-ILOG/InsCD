import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .._base import _Extractor
import random
import scipy.sparse as sp
import dgl

class GraphLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GraphLayer, self).__init__()
        self.g = g
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

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class Fusion(nn.Module):
    def __init__(self, args, local_map):
        self.device = args['device']
        self.knowledge_dim = args['knowledge_num']
        self.exer_n = args['exercise_num']
        self.emb_num = args['student_num']
        self.stu_dim = args['knowledge_num']
        # data structure
        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        self.k_from_e = local_map['k_from_e'].to(self.device)

        self.e_from_k = local_map['e_from_k'].to(self.device)


        super(Fusion, self).__init__()

        self.directed_gat = GraphLayer(self.directed_g, self.knowledge_dim, self.knowledge_dim)
        self.undirected_gat = GraphLayer(self.undirected_g, self.knowledge_dim, self.knowledge_dim)
        self.e_from_k = GraphLayer(self.e_from_k, self.knowledge_dim, self.knowledge_dim)  # src: k

        self.k_from_e = GraphLayer(self.k_from_e, self.knowledge_dim, self.knowledge_dim)  # src: e

        self.k_attn_fc1 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.k_attn_fc2 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.k_attn_fc3 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)

        self.e_attn_fc1 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)

    def forward(self, exer_emb,kn_emb):
        k_directed = self.directed_gat(kn_emb)  
        k_undirected = self.undirected_gat(kn_emb)

        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        e_from_k_graph = self.e_from_k(e_k_graph)
        # update concepts
        A = kn_emb
        B = k_directed
        C = k_undirected
        concat_c_1 = torch.cat([A, B], dim=1)
        concat_c_2 = torch.cat([A, C], dim=1)
        score1 = self.k_attn_fc1(concat_c_1) 
        score2 = self.k_attn_fc2(concat_c_2)  
        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)
                         
        kn_emb = A + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C


        # updated exercises
        A = exer_emb
        B = e_from_k_graph[0:self.exer_n]
        concat_e_1 = torch.cat([A, B], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        exer_emb = exer_emb + score1[:, 0].unsqueeze(1) * B

        return exer_emb,kn_emb


class Op(nn.Module):
    '''
    operation for one link in the DAG search space
    '''

    def __init__(self, k):
        super(Op, self).__init__()
        self.k = k   # nc: k=1   lr: k=2


    def forward(self, x, adjs, ws):
        num_op = len(ws)
        num = int(num_op//self.k)
        idx = random.sample(range(num_op),num)
        return sum(ws[i] * torch.spmm(adjs[i], x) for i in range(num_op) if i in idx) / num #self.k  #num   #self.k


class Cell(nn.Module):
    '''
    the DAG search space
    '''
    def __init__(self, n_step, n_hid_prev, n_hid, cstr, k, use_norm = True, use_nl = True, ratio = 1):
        super(Cell, self).__init__()
        
        self.affine = nn.Linear(n_hid_prev, n_hid)
        self.n_step = n_step               #* number of intermediate states (i.e., K)
        self.norm = nn.LayerNorm(n_hid, elementwise_affine = False) if use_norm is True else lambda x : x
        self.use_nl = use_nl
        assert(isinstance(cstr, list))
        self.cstr = cstr                   #* type constraint
        self.ratio = ratio
        op = Op(k)

        self.ops_seq = nn.ModuleList()     #* state (i - 1) -> state i, 1 <= i < K,  AI,  seq: sequential
        for i in range(1, self.n_step):
            self.ops_seq.append(op)
        self.ops_res = nn.ModuleList()     #* state j -> state i, 0 <= j < i - 1, 2 <= i < K,  AIO,  res: residual
        for i in range(2, self.n_step):
            for j in range(i - 1):
                self.ops_res.append(op)

        self.last_seq = op               #* state (K - 1) -> state K,  /hat{A}
        self.last_res = nn.ModuleList()    #* state i -> state K, 0 <= i < K - 1,  /hat{A}IO
        for i in range(self.n_step - 1):
            self.last_res.append(op)


    

    def forward(self, x, adjs, ws_seq, ws_res):
        #assert(isinstance(ws_seq, list))
        #assert(len(ws_seq) == 2)

        x = self.affine(x)
        states = [x]
        offset = 0
        edge = 1
        for i in range(self.n_step - 1):
            seqi = self.ops_seq[i](states[i], adjs[:-1], ws_seq[0][i])   #! exclude zero Op
            resi = sum(self.ops_res[offset + j](h, adjs, ws_res[0][offset + j]) for j, h in enumerate(states[:i]))
            offset += i
            states.append((seqi + self.ratio * resi)/edge)
        #assert(offset == len(self.ops_res))

        adjs_cstr = [adjs[i] for i in self.cstr]
        out_seq = self.last_seq(states[-1], adjs_cstr, ws_seq[1])

        adjs_cstr.append(adjs[-1])
        out_res = sum(self.last_res[i](h, adjs_cstr, ws_res[1][i]) for i, h in enumerate(states[:-1]))
        output = self.norm((out_seq + self.ratio * out_res)/edge)
        if self.use_nl:
            output = F.gelu(output)
        return output


class Model_paths(nn.Module):

    def __init__(self,gpu, in_dim, n_hid, num_node_types, n_adjs, n_classes, n_steps, ratio, cstr, k, lambda_seq, lambda_res, attn_dim = 64, use_norm = True, out_nl = True):
        super(Model_paths, self).__init__()
        self.device = gpu
        self.num_node_types = num_node_types
        self.cstr = cstr  
        self.n_adjs = n_adjs  
        self.n_hid = n_hid   
        self.ws = nn.ModuleList()          #* node type-specific transformation
        self.lambda_seq = lambda_seq
        self.lambda_res = lambda_res
        for i in range(num_node_types): 
            self.ws.append(nn.Linear(in_dim, n_hid))  
        assert(isinstance(n_steps, list))  #* [optional] combine more than one meta data?
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):  
            self.metas.append(Cell(n_steps[i], n_hid, n_hid, cstr, k, use_norm = use_norm, use_nl = out_nl, ratio = ratio))  # self.metas contions 1 Cell

        self.as_seq = []                   #* arch parameters for ops_seq    k<K and i=k-1   AI
        self.as_last_seq = []              #* arch parameters for last_seq   k=K and i=k-1  /hat{A}
        for i in range(len(n_steps)):
            if n_steps[i] > 1:  # not for
                ai = 1e-3 * torch.randn(n_steps[i] - 1, (n_adjs - 1))   #! exclude zero Op   torch.randn(3, 5)  AI
                ai = ai.to(self.device)
                ai.requires_grad_(True)
                self.as_seq.append(ai)
            else:
                self.as_seq.append(None)
            ai_last = 1e-3 * torch.randn(len(cstr))  # torch.randn(2)  edge related to the evaluation  /hat{A}   actually /hat{A} I
            ai_last = ai_last.to(self.device)
            ai_last.requires_grad_(True)
            self.as_last_seq.append(ai_last)

        ks = [sum(1 for i in range(2, n_steps[k]) for j in range(i - 1)) for k in range(len(n_steps))]
        self.as_res = []                  #* arch parameters for ops_res    k<K and i<k-1    AIO
        self.as_last_res = []             #* arch parameters for last_res   k=K and i<k-1    /hat{A}IO
        for i in range(len(n_steps)):
            if ks[i] > 0:
                ai = 1e-3 * torch.randn(ks[i], n_adjs)  # (3,6)  AIO
                ai = ai.to(self.device)
                ai.requires_grad_(True)
                self.as_res.append(ai)
            else:
                self.as_res.append(None)
            
            if n_steps[i] > 1:
                ai_last = 1e-3 * torch.randn(n_steps[i] - 1, len(cstr) + 1) 
                ai_last = ai_last.to(self.device)
                ai_last.requires_grad_(True)
                self.as_last_res.append(ai_last)
            else:
                self.as_last_res.append(None)

        assert(ks[0] + n_steps[0] + (0 if self.as_last_res[0] is None else self.as_last_res[0].size(0)) == (1 + n_steps[0]) * n_steps[0] // 2)


        #* [optional] combine more than one meta data?
        self.attn_fc1 = nn.Linear(n_hid, attn_dim) 
        self.attn_fc2 = nn.Linear(attn_dim, 1)  

        self.classifier = nn.Linear(n_hid, n_classes)

    def forward(self, node_feats, node_types, adjs):
        hid = torch.zeros((node_types.size(0), self.n_hid)).to(self.device)
        for i in range(self.num_node_types):
            idx = (node_types == i)
            hid[idx] = self.ws[i](node_feats[idx])
        temps = []
        attns = []
        for i, meta in enumerate(self.metas):
            ws_seq = []
            ws_seq.append(None if self.as_seq[i] is None else F.softmax(self.as_seq[i], dim=-1))  # softmax here
            ws_seq.append(F.softmax(self.as_last_seq[i], dim=-1)) 
            ws_res = []
            ws_res.append(None if self.as_res[i] is None else F.softmax(self.as_res[i], dim=-1))
            ws_res.append(None if self.as_last_res[i] is None else F.softmax(self.as_last_res[i], dim=-1))
            hidi = meta(hid, adjs, ws_seq, ws_res)  # cell
            temps.append(hidi)  
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1]))) # attni.shape   
            attns.append(attni)

        hids = torch.stack(temps, dim=0).transpose(0, 1)  
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1) 
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)  # attns.unsqueeze(dim=-1) * hids 
        logits = self.classifier(out) 
        return logits


    def alphas(self):
        alphas = []
        for each in self.as_seq:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_seq:
            alphas.append(each)
        for each in self.as_res:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_res:
            if each is not None:
                alphas.append(each)

        return alphas

    def getid(self, seq_res, lam):
        seq_softmax = None if seq_res is None else F.softmax(seq_res, dim=-1)

        length = seq_res.size(-1)
        if len(seq_res.shape) == 1:
            max = torch.max(seq_softmax, dim=0).values
            min = torch.min(seq_softmax, dim=0).values
            threshold = lam * max + (1 - lam) * min
            return [k for k in range(length) if seq_softmax[k].item()>=threshold]
        max = torch.max(seq_softmax, dim=1).values
        min = torch.min(seq_softmax, dim=1).values
        threshold = lam * max + (1 - lam) * min
        res = [[k for k in range(length) if seq_softmax[j][k].item() >= threshold[j]] for j in range(len(seq_softmax))]
        return res

    def sample_final(self, eps):
        '''
        to sample one candidate edge type per link
        '''
        idxes_seq = []
        idxes_res = []
        if np.random.uniform() < eps:
            for i in range(len(self.metas)): 
                temp = []
                temp.append(None if self.as_seq[i] is None else torch.randint(low=0, high=self.as_seq[i].size(-1), size=self.as_seq[i].size()[:-1]).to(self.device))
                temp.append(torch.randint(low=0, high=self.as_last_seq[i].size(-1), size=(1,)).to(self.device))
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else torch.randint(low=0, high=self.as_res[i].size(-1), size=self.as_res[i].size()[:-1]).to(self.device))  # self.as_res[0]: shape [3,6]   high:  6   size :3
                temp.append(None if self.as_last_res[i] is None else torch.randint(low=0, high=self.as_last_res[i].size(-1), size=self.as_last_res[i].size()[:-1]).to(self.device)) # self.as_last_res[0]: shape [3,3]   high:  3   size :3
                idxes_res.append(temp)
        else:
            for i in range(len(self.metas)):
                temp = []
                seq = self.getid(self.as_seq[i], self.lambda_seq)
                last_seq = self.getid(self.as_last_seq[i], self.lambda_seq)
                temp.append(seq)
                temp.append(last_seq)
                idxes_seq.append(temp)

            for i in range(len(self.metas)):
                temp = []
                res = self.getid(self.as_res[i], self.lambda_res)
                last_res = self.getid(self.as_last_res[i], self.lambda_res)
                temp.append(res)
                temp.append(last_res)
                idxes_res.append(temp)
        return idxes_seq, idxes_res

    
    def parse(self):
        '''
        to derive a meta data indicated by arch parameters
        '''
        idxes_seq, idxes_res = self.sample_final(0.)

        msg_seq = []; msg_res = []
        for i in range(len(idxes_seq)):
            map_seq = [[self.cstr[item] for item in idxes_seq[i][1]]]
            msg_seq.append(map_seq if idxes_seq[i][0] is None else idxes_seq[i][0] + map_seq) #idxes_seq[0][0]+idxes_seq[0][1]

            assert(len(msg_seq[i]) == self.metas[i].n_step)
            temp_res = []
            if idxes_res[i][1] is not None:
                for res in idxes_res[i][1]:
                    temp = []
                    for item in res:
                        if item < len(self.cstr):
                            temp.append(self.cstr[item])
                        else:
                            assert(item == len(self.cstr))
                            temp.append(self.n_adjs - 1)
                    temp_res.append(temp)
                if idxes_res[i][0] is not None:
                    temp_res = idxes_res[i][0] + temp_res   # idxes_res[0][0]+idxes_res[0][1]
            assert(len(temp_res) == self.metas[i].n_step * (self.metas[i].n_step - 1) // 2)
            msg_res.append(temp_res)
        

        return msg_seq, msg_res
    
def build_graph(config,type, node):
    ek=torch.from_numpy(config['datahub'].q_matrix)

    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    if type == 'direct':
        if config['direct_path']:
            with open(config['direct_path'], 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '').split('\t')
                    edge_list.append((int(line[0]), int(line[1])))
            edge_list = list(set(edge_list))
            src, dst = tuple(zip(*edge_list))
            g.add_edges(src, dst)
            return g
        else:
            di_kk=torch.ones((config["knowledge_num"],config["knowledge_num"]))-np.identity(config["knowledge_num"])
            edge_index = (di_kk == 1).nonzero(as_tuple=False)  # shape: [num_edges, 2]
            src, dst = edge_index[:, 0], edge_index[:, 1]
            g.add_edges(src, dst)
            return g
    elif type == 'undirect':
        if config['undirect_path']:
            with open(config['undirect_path'], 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '').split('\t')
                    edge_list.append((int(line[0]), int(line[1])))
            edge_list = list(set(edge_list))
            src, dst = tuple(zip(*edge_list))
            g.add_edges(src, dst)
            g.add_edges(dst, src)
            return g
        else:
            un_kk=torch.ones((config["knowledge_num"],config["knowledge_num"]))-np.identity(config["knowledge_num"])
            edge_index = (un_kk == 1).nonzero(as_tuple=False)  # shape: [num_edges, 2]
            src, dst = edge_index[:, 0], edge_index[:, 1]
            g.add_edges(src, dst)
            g.add_edges(dst, src)
            return g
    elif type == 'k_from_e':
        edge_index = (ek == 1).nonzero(as_tuple=False)  # shape: [num_edges, 2]
        src, dst = edge_index[:, 0], edge_index[:, 1]
        g.add_edges(src, dst)
        return g

    elif type == 'e_from_k':
        edge_index = (ek == 1).nonzero(as_tuple=False)  # shape: [num_edges, 2]
        src, dst = edge_index[:, 0], edge_index[:, 1]
        g.add_edges(dst, src)
        return g

class DisenGCD_EX(_Extractor, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.student_num = config['student_num']
        self.exercise_num = config['exercise_num']
        self.knowledge_num = config['knowledge_num']
        self.device = config['device']
        self.latent_dim = config['latent_dim']
        local_map=self.construct_local_map(config)
        self.map=self.construct_map()
        self.node_type=torch.cat([
            torch.zeros(self.student_num, dtype=torch.long),
            torch.ones(self.exercise_num, dtype=torch.long),
            torch.full((self.knowledge_num,), 2, dtype=torch.long)
        ])
        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)


        self.stu_emb = nn.Embedding(self.student_num,self.knowledge_num).to(self.device)
        self.kn_emb = nn.Embedding(self.knowledge_num, self.knowledge_num)  
        self.exer_emb = nn.Embedding(self.exercise_num, self.knowledge_num)  

        self.index = torch.LongTensor(list(range(self.student_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exercise_num))).to(self.device)  
        self.k_index = torch.LongTensor(list(range( self.knowledge_num))).to(self.device)  

        self.FusionLayer1 = Model_paths(self.device,self.knowledge_num,self.latent_dim,3,config["len_map"],self.knowledge_num, [4],
                                  config['ratio'],[3,4],1,config["lam_seq"],config["lam_res"])                         
        self.FusionLayer3 = Fusion(config, local_map)
        self.FusionLayer4 = Fusion(config, local_map)

        self.prednet_full3 = nn.Linear(self.knowledge_num, 1)
        self.disc_generate=nn.Linear(self.knowledge_num,1)

        self.__emb_map = {
            "mastery": self.stu_emb.weight,
            "diff": self.exer_emb.weight,
            "knowledge": self.kn_emb.weight
        }

        self.apply(self.initialize_weights)


    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def extract(self, student_id, exercise_id, q_mask):
        all_stu_emb = self.stu_emb(self.index).to(self.device)
        exer_emb = self.exer_emb(self.exer_index).to(self.device)
        kn_emb = self.kn_emb(self.k_index).to(self.device)

  
        all_emb = torch.cat((all_stu_emb,exer_emb,kn_emb),0)
        all_stu_emb1 = self.FusionLayer1(all_emb, self.node_type, self.map)
        all_stu_emb1 = all_stu_emb1[self.knowledge_num+self.exercise_num:self.student_num + self.exercise_num+self.student_num, :]
        #all_stu_emb2 = all_stu_emb1[1747:3714, :]

        exer_emb1,kn_emb1 = self.FusionLayer3(exer_emb,kn_emb)
        exer_emb2,knowledge_ts = self.FusionLayer4(exer_emb1,kn_emb1)
        disc_ts=self.disc_generate(exer_emb2)

        diff_ts = exer_emb2[exercise_id]
        disc_ts=disc_ts[exercise_id]
        
        student_ts = all_stu_emb1[student_id]  # 32 123
       
        return student_ts, diff_ts, disc_ts, knowledge_ts,{}
    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        all_stu_emb = self.stu_emb(self.index).to(self.device)
        exer_emb = self.exer_emb(self.exer_index).to(self.device)
        kn_emb = self.kn_emb(self.k_index).to(self.device)

  
        all_emb = torch.cat((all_stu_emb,exer_emb,kn_emb),0)
        all_stu_emb1 = self.FusionLayer1(all_emb, self.node_type, self.map)
        student_ts = all_stu_emb1[self.knowledge_num+self.exercise_num:self.student_num + self.exercise_num+self.student_num, :]

        exer_emb1,kn_emb1 = self.FusionLayer3(exer_emb,kn_emb)
        diff_ts,knowledge_ts = self.FusionLayer4(exer_emb1,kn_emb1)
        disc_ts=self.disc_generate(diff_ts)

        self.__emb_map["mastery"] = student_ts
        self.__emb_map["diff"] = diff_ts
        self.__emb_map["disc"] = disc_ts
        self.__emb_map["knowledge"] = knowledge_ts
        return self.__emb_map[item]
    
    def construct_map(self):
        se = self.get_interact_mat()
        ek = self.config['datahub'].q_matrix
        kk=torch.ones((self.knowledge_num,self.knowledge_num))-np.identity(self.knowledge_num)
        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp =[np.zeros(shape=(sek_num, sek_num)) for i in range(6)]
        tmp[0][:self.student_num, self.student_num: se_num] = se
        tmp[0]=tmp[0]+tmp[0].T+np.identity(sek_num)
        tmp[1][self.student_num:se_num, se_num:sek_num] = ek
        tmp[1]=tmp[1]+tmp[1].T+np.identity(sek_num)
        tmp[2][se_num:, se_num:] = kk
        tmp[2]=tmp[2]+np.identity(sek_num)
        tmp[3][se_num:, se_num:] = kk
        tmp[3]=tmp[3]+np.identity(sek_num)
        tmp[4]=tmp[4]+np.identity(sek_num)
        adjs_pt = []

        for i in range(len(tmp)):
            mx = sp.coo_matrix(tmp[i])
            if i<4:
                mx = normalize_row(mx)
            mx_tensor = sparse_mx_to_torch_sparse_tensor(mx)
            adjs_pt.append(mx_tensor.to(self.config["device"]))
        return adjs_pt
    
    def construct_local_map(self,args):
        local_map = {
            'directed_g': build_graph(args,'direct',args['knowledge_num']),
            'undirected_g': build_graph(args,'undirect', args['knowledge_num']),
            'k_from_e': build_graph(args,'k_from_e', args['knowledge_num'] + args['exercise_num']),
            'e_from_k': build_graph(args,'e_from_k', args['knowledge_num'] + args['exercise_num']),
        }
        return local_map
    
        
    def get_interact_mat(self):
        interact_mat = torch.zeros((self.config['student_num'], self.config['exercise_num']))
        for row in range(self.config['datahub']['train'].shape[0]):
            interact_mat[int(self.config['datahub']['train'][row, 0]), int(self.config['datahub']['train'][row, 1])] = 1
        return interact_mat
    
def normalize_row(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


