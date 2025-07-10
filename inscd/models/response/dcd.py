from typing import Union
from tqdm import tqdm
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from edustudio.model import GDBaseModel
from torch.autograd import Function

from ..._base import _CognitiveDiagnosisModel
from ...datahub import DataHub


class STHeaviside(Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.zeros(x.size()).type_as(x)
        y[x >= 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class MLP(nn.Module):
    """
        The Multi Layer Perceptron (MLP)
        note: output layer has no activation function, output layer has batch norm and dropout
    """

    def __init__(self, input_dim: int, output_dim: int, dnn_units: Union[list, tuple],
                 activation: Union[str, nn.Module, list] = 'relu', dropout_rate: float = 0.0,
                 use_bn: bool = False, device='cpu'):
        super().__init__()
        self.use_bn = use_bn
        dims_list = [input_dim] + list(dnn_units) + [output_dim]
        if type(activation) is list:
            assert len(activation) == len(dnn_units)

        self.linear_units_list = nn.ModuleList(
            [nn.Linear(dims_list[i], dims_list[i + 1], bias=True) for i in range(len(dims_list) - 1)]
        )
        self.act_units_list = nn.ModuleList(
            [ActivationUtil.get_common_activation_layer(activation)] * len(dnn_units)
            if type(activation) is not list else [ActivationUtil.get_common_activation_layer(i) for i in activation]
        )
        self.dropout_layer = nn.Dropout(dropout_rate)

        if use_bn is True:
            self.bn_units_list = nn.ModuleList(
                [nn.BatchNorm1d(dims_list[i + 1]) for i in range(len(dims_list) - 1)]
            )
            assert len(self.linear_units_list) == len(self.bn_units_list)
        assert len(self.linear_units_list) == len(self.act_units_list) + 1
        self.to(device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tmp = input
        for i in range(len(self.act_units_list)):
            tmp = self.linear_units_list[i](tmp)
            if self.use_bn is True:
                tmp = self.bn_units_list[i](tmp)
            tmp = self.act_units_list[i](tmp)
            tmp = self.dropout_layer(tmp)
        tmp = self.linear_units_list[-1](tmp)
        if self.use_bn is True:
            tmp = self.bn_units_list[-1](tmp)
        output = tmp
        return output


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class ActivationUtil(object):
    @staticmethod
    def get_common_activation_layer(act_obj: Union[str, nn.Module] = "relu") -> nn.Module:
        if isinstance(act_obj, str):
            if act_obj.lower() == 'relu':
                return nn.ReLU(inplace=True)
            elif act_obj.lower() == 'sigmoid':
                return nn.Sigmoid()
            elif act_obj.lower() == 'linear':
                return Identity()
            elif act_obj.lower() == 'prelu':
                return nn.PReLU()
            elif act_obj.lower() == 'elu':
                return nn.ELU(inplace=True)
            elif act_obj.lower() == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)
        else:
            return act_obj()


class PosMLP(nn.Module):
    """
        The Multi Layer Perceptron (MLP)
        note: output layer has no activation function, output layer has batch norm and dropout
    """

    def __init__(self, input_dim: int, output_dim: int, dnn_units: Union[list, tuple],
                 activation: Union[str, nn.Module, list] = 'relu', dropout_rate: float = 0.0,
                 use_bn: bool = False, device='cpu'):
        super().__init__()
        self.use_bn = use_bn
        dims_list = [input_dim] + list(dnn_units) + [output_dim]
        if type(activation) is list:
            assert len(activation) == len(dnn_units)

        self.linear_units_list = nn.ModuleList(
            [PosLinear(dims_list[i], dims_list[i + 1], bias=True) for i in range(len(dims_list) - 1)]
        )
        self.act_units_list = nn.ModuleList(
            [ActivationUtil.get_common_activation_layer(activation)] * len(dnn_units)
            if type(activation) is not list else [ActivationUtil.get_common_activation_layer(i) for i in activation]
        )
        self.dropout_layer = nn.Dropout(dropout_rate)

        if use_bn is True:
            self.bn_units_list = nn.ModuleList(
                [nn.BatchNorm1d(dims_list[i + 1]) for i in range(len(dims_list) - 1)]
            )
            assert len(self.linear_units_list) == len(self.bn_units_list)
        assert len(self.linear_units_list) == len(self.act_units_list) + 1
        self.to(device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tmp = input
        for i in range(len(self.act_units_list)):
            tmp = self.linear_units_list[i](tmp)
            if self.use_bn is True:
                tmp = self.bn_units_list[i](tmp)
            tmp = self.act_units_list[i](tmp)
            tmp = self.dropout_layer(tmp)
        tmp = self.linear_units_list[-1](tmp)
        if self.use_bn is True:
            tmp = self.bn_units_list[-1](tmp)
        output = tmp
        return output


class MarginLossZeroOne(nn.Module):
    def __init__(self, margin=0.5, reduction: str = 'mean') -> None:
        assert reduction in ['mean', 'sum', 'none']
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, pos_pd, neg_pd):
        logits = self.margin - (pos_pd - neg_pd)
        logits[logits < 0] = 0.0
        if self.reduction == 'mean':
            return logits.mean()
        elif self.reduction == 'sum':
            return logits.sum()
        else:
            return logits


class NormalDistUtil(object):
    @staticmethod
    def log_density(X, MU, LOGVAR):
        """ compute log pdf of normal distribution

        Args:
            X (_type_): sample point
            MU (_type_): mu of normal dist
            LOGVAR (_type_): logvar of normal dist
        """
        norm = - 0.5 * (math.log(2 * math.pi) + LOGVAR)
        log_density = norm - 0.5 * ((X - MU).pow(2) * torch.exp(-LOGVAR))
        return log_density

    @staticmethod
    def kld(MU: float, LOGVAR: float, mu_move):
        """compute KL divergence between X and Normal Dist whose (mu, var) equals to (mu_move, 1)

        Args:
            MU (float): _description_
            VAR (float): _description_
            mu_move (_type_): _description_
        """

        return 0.5 * (LOGVAR.exp() - LOGVAR + MU.pow(2) - 2 * mu_move * MU + mu_move ** 2 - 1)

    @staticmethod
    def sample(mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps


eps = 1e-8


class BernoulliUtil(nn.Module):
    """Samples from a Bernoulli distribution where the probability is given
    by the sigmoid of the given parameter.
    """

    def __init__(self, p=0.5, stgradient=False):
        super().__init__()
        p = torch.Tensor([p])
        self.p = torch.log(p / (1 - p) + eps)
        self.stgradient = stgradient

    def _check_inputs(self, size, ps):
        if size is None and ps is None:
            raise ValueError(
                'Either one of size or params should be provided.')
        elif size is not None and ps is not None:
            if ps.ndimension() > len(size):
                return ps.squeeze(-1).expand(size)
            else:
                return ps.expand(size)
        elif size is not None:
            return self.p.expand(size)
        elif ps is not None:
            return ps
        else:
            raise ValueError(
                'Given invalid inputs: size={}, ps={})'.format(size, ps))

    def _sample_logistic(self, size):
        u = torch.rand(size)
        l = torch.log(u + eps) - torch.log(1 - u + eps)
        return l

    def default_sample(self, size=None, params=None):
        presigm_ps = self._check_inputs(size, params)
        logp = F.logsigmoid(presigm_ps)
        logq = F.logsigmoid(-presigm_ps)
        l = self._sample_logistic(logp.size()).type_as(presigm_ps)
        z = logp - logq + l
        b = STHeaviside.apply(z)
        return b if self.stgradient else b.detach()

    def sample(self, size=None, params=None, type_='gumbel_softmax', **kwargs):
        if type_ == 'default':
            return self.default_sample(size, params)
        elif type_ == 'gumbel_softmax':
            tau = kwargs.get('tau', 1.0)
            hard = kwargs.get('hard', True)
            ext_params = torch.log(torch.stack([1 - params, params], dim=2) + eps)
            return F.gumbel_softmax(logits=ext_params, tau=tau, hard=hard)[:, :, -1]
        else:
            raise ValueError(f"Unknown Type of sample: {type_}")

    def log_density(self, sample, params=None, is_check=True):
        if is_check:
            presigm_ps = self._check_inputs(sample.size(), params).type_as(sample)
        else:
            presigm_ps = params
        p = (torch.sigmoid(presigm_ps) + eps) * (1 - 2 * eps)
        logp = sample * torch.log(p + eps) + (1 - sample) * torch.log(1 - p + eps)
        return logp

    def get_params(self):
        return self.p

    @property
    def nparams(self):
        return 1

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return self.stgradient

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' ({:.3f})'.format(
            torch.sigmoid(self.p.data)[0])
        return tmpstr


class DCDModules(nn.Module):
    def __init__(self, config):
        super(DCDModules, self).__init__()
        exercise_num = config['exercise_num']
        student_num = config['student_num']
        knowledge_num = config['knowledge_num']
        self.EncoderStudent = MLP(
            input_dim=exercise_num,
            output_dim=knowledge_num * 2,
            dnn_units=config['EncoderStudentHidden']
        )
        self.EncoderExercise = MLP(
            input_dim=student_num,
            output_dim=knowledge_num,
            dnn_units=config['EncoderExerciseHidden']
        )

        self.EncoderExerciseDiff = MLP(
            input_dim=student_num,
            output_dim=knowledge_num * 2,
            dnn_units=config['EncoderExerciseHidden']
        )
        self.ExerciseDisc = nn.Embedding(exercise_num, 1)
        self.pd_net = PosMLP(
            input_dim=knowledge_num, output_dim=1, activation=config['pred_activation'],
            dnn_units=config['pred_dnn_units'], dropout_rate=config['pred_dropout_rate']
        )


class DCD(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        DCD: Disentangling cognitive diagnosis
        Xiangzhi Chen et al. Disentangling cognitive diagnosis with limited exercise labels. NeurIPS'2023.

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
        self.dcdModules = DCDModules(config)
        self.margin_loss_zero_one = MarginLossZeroOne(reduction='none',
                                                      margin=self.config['align_margin_loss_kwargs']['margin'])

        self.student_dist = NormalDistUtil()
        self.exercise_dist = BernoulliUtil(p=self.config['bernoulli_prior_p'], stgradient=True)
        self.exercise_dist_diff = NormalDistUtil()
        self.get_interact_mat()
        self.get_q_mat()

        
    def get_interact_mat(self):
        interact_mat = torch.zeros((self.config['student_num'], self.config['exercise_num']))
        for row in range(self.config['datahub']['train'].shape[0]):
            if int(self.config['datahub']['train'][row, 2]) == 0:
                interact_mat[int(self.config['datahub']['train'][row, 0]), int(self.config['datahub']['train'][row, 1])] = -1
            else:
                interact_mat[int(self.config['datahub']['train'][row, 0]), int(self.config['datahub']['train'][row, 1])] = 1
        self.interact_mat = interact_mat
    
    def get_q_mat(self):
        self.Q_mat = torch.from_numpy(self.config['datahub'].q_matrix)



    def get_pred(self, student_id, exercise_id):
        student_mix = self.dcdModules.EncoderStudent(self.interact_mat[student_id, :])
        student_emb, _ = torch.chunk(student_mix, 2, dim=-1)
        # student_emb = self.student_dist.sample(student_mu, student_logvar)
        exercise_emb = self.dcdModules.EncoderExercise(self.interact_mat[:, exercise_id].T).sigmoid()
        # exercise_emb = self.exercise_dist.sample(None, exercise_mu, type_=self.config['b_sample_type'],**self.config['b_sample_kwargs'])
        exercise_emb_diff_mix = self.dcdModules.EncoderExerciseDiff(self.interact_mat[:, exercise_id].T)
        exercise_emb_diff, _ = torch.chunk(exercise_emb_diff_mix, 2, dim=-1)
        # exercise_emb_diff = self.exercise_dist_diff.sample(exercise_mu_diff, exercise_logvar_diff)
        return self.decode(student_emb, exercise_emb, exercise_emb_diff, exercise_id=exercise_id).sigmoid()
        
    def get_align_exercise_loss(self, exercise_emb, exercise_idx):
        if self.config['align_type'] == 'mse_margin':
            self.Q_mat = self.Q_mat.to(exercise_emb.device)
            flag = self.Q_mat[exercise_idx, :].sum(dim=1) > 0
            left_emb = exercise_emb[~flag]
            p = self.config['align_margin_loss_kwargs']['norm']
            t_loss = torch.norm(left_emb, dim=0, p=p).pow(p).sum()
            if left_emb.shape[0] != 0 and self.callback_list.curr_epoch >= \
                    self.config['align_margin_loss_kwargs']['start_epoch']:
                topk_idx = torch.topk(left_emb, self.config['align_margin_loss_kwargs']['topk'] + 1).indices
                pos = torch.gather(left_emb, 1, topk_idx[:, 0:self.config['align_margin_loss_kwargs']['d1']])
                neg = torch.gather(left_emb, 1, topk_idx[:, [-1]])
                margin_loss = self.margin_loss_zero_one(pos, neg).mean(dim=1).sum()
            else:
                margin_loss = torch.tensor(0.0).to(exercise_emb.device)
            return {
                "mse_loss": F.mse_loss(exercise_emb[flag], self.Q_mat[exercise_idx[flag], :].float(), reduction='sum'),
                "margin_loss": margin_loss,
                "norm_loss": t_loss,
            }
        elif self.config['align_type'] == 'mse_margin_mean':
            flag = self.Q_mat[exercise_idx, :].sum(dim=1) > 0
            left_emb = exercise_emb[~flag]
            p = self.config['align_margin_loss_kwargs']['norm']
            t_loss = torch.norm(left_emb, dim=0, p=p).pow(p).sum()
            if left_emb.shape[0] != 0 and self.callback_list.curr_epoch >= \
                    self.config['align_margin_loss_kwargs']['start_epoch']:
                # topk_idx = torch.topk(left_emb, self.config['align_margin_loss_kwargs']['topk']).indices
                # bottomk_idx = torch.ones_like(left_emb).scatter(1, topk_idx, 0).nonzero()[:, 1].reshape(-1, left_emb.size(1) - topk_idx.size(1))
                # pos = torch.gather(left_emb, 1, topk_idx[:,[-1]])
                # neg = torch.gather(left_emb, 1, bottomk_idx[:,torch.randperm(bottomk_idx.shape[1],dtype=torch.long)[0:int(bottomk_idx.shape[1]*0.5)]])
                topk_idx = torch.topk(left_emb, self.config['align_margin_loss_kwargs']['topk'] + 1).indices
                bottomk_idx = torch.topk(-left_emb, left_emb.shape[1] - self.config['align_margin_loss_kwargs'][
                    'topk']).indices
                pos = torch.gather(left_emb, 1, topk_idx[:, 0:self.config['align_margin_loss_kwargs']['d1']])
                neg = torch.gather(left_emb, 1, bottomk_idx).mean(dim=1)
                margin_loss = self.margin_loss_zero_one(pos, neg).mean(dim=1).sum()
            else:
                margin_loss = torch.tensor(0.0).to(exercise_emb.device)
            return {
                "mse_loss": F.mse_loss(exercise_emb[flag], self.Q_mat[exercise_idx[flag], :].float(), reduction='sum'),
                "margin_loss": margin_loss,
                "norm_loss": t_loss,
            }
        else:
            raise ValueError(f"Unknown align type: {self.config['align_type']}")

    def decode(self, student_emb, exercise_emb, exercise_emb_diff, exercise_id, **kwargs):
        if self.config['interact_type'] == 'irt_wo_disc':
            return ((student_emb - exercise_emb_diff) * exercise_emb).sum(dim=1)
        elif self.config['interact_type'] == 'irt':
            exercise_disc = self.dcdModules.ExerciseDisc(exercise_id).sigmoid()  # * self.config['disc_scale']
            return ((student_emb - exercise_emb_diff) * exercise_emb * exercise_disc).sum(dim=1)
        elif self.config['interact_type'] == 'ncdm':
            exercise_disc = self.dcdModules.ExerciseDisc(exercise_id).sigmoid()  # * self.config['disc_scale']
            input = (student_emb - exercise_emb_diff) * exercise_emb * exercise_disc
            return self.dcdModules.pd_net(input).flatten()
        elif self.config['interact_type'] == 'mf':
            return ((student_emb.sigmoid() * exercise_emb) * (exercise_emb * exercise_emb_diff)).sum(dim=1)
        elif self.config['interact_type'] == 'mirt':  # 就是mf加了个disc
            exercise_disc = self.dcdModules.ExerciseDisc(exercise_id).sigmoid()  # * self.config['disc_scale']
            return ((student_emb.sigmoid() * exercise_emb) * (exercise_emb * exercise_emb_diff)).sum(
                dim=1) + exercise_disc.flatten()
        else:
            raise NotImplementedError

    def forward(self, students, exercises, labels):
        student_unique, student_unique_idx = students.unique(sorted=True, return_inverse=True)
        exercise_unique, exercise_unique_idx = exercises.unique(sorted=True, return_inverse=True)

        self.interact_mat = self.interact_mat.to(student_unique.device)
        student_mix = self.dcdModules.EncoderStudent(self.interact_mat[student_unique, :])
        student_mu, student_logvar = torch.chunk(student_mix, 2, dim=-1)
        student_emb_ = self.student_dist.sample(student_mu, student_logvar)
        student_emb = student_emb_[student_unique_idx, :]

        exercise_mu = self.dcdModules.EncoderExercise(self.interact_mat[:, exercise_unique].T).sigmoid()
        exercise_emb_ = self.exercise_dist.sample(None, exercise_mu, type_=self.config['b_sample_type'],
                                                  **self.config['b_sample_kwargs'])
        exercise_emb = exercise_emb_[exercise_unique_idx, :]

        exercise_diff_mix = self.dcdModules.EncoderExerciseDiff(self.interact_mat[:, exercise_unique].T)
        exercise_mu_diff, exercise_logvar_diff = torch.chunk(exercise_diff_mix, 2, dim=-1)
        exercise_emb_diff_ = self.exercise_dist_diff.sample(exercise_mu_diff, exercise_logvar_diff)
        exercise_emb_diff = exercise_emb_diff_[exercise_unique_idx, :]

        loss_main = F.binary_cross_entropy_with_logits(
            self.decode(student_emb, exercise_emb, exercise_emb_diff, exercise_id=exercises),
            labels, reduction='sum')  # 重构 loss
        align_loss_dict = self.get_align_exercise_loss(exercise_mu, exercise_unique)
        # align_loss_dict_diff = self.get_align_exercise_loss(exercise_mu_diff, exercise_unique)

        student_terms = self.get_tcvae_terms(student_emb_, params=(student_mu, student_logvar), dist=self.student_dist,
                                             dataset_size=self.config['student_num'])
        exercise_terms = self.get_tcvae_terms(exercise_emb_, params=exercise_mu, dist=self.exercise_dist,
                                              dataset_size=self.config['exercise_num'])
        exercise_terms_diff = self.get_tcvae_terms(exercise_emb_diff_, params=(exercise_mu_diff, exercise_logvar_diff),
                                                   dist=self.exercise_dist_diff, dataset_size=self.config['exercise_num'])

        return {
            'loss_main': loss_main * self.config['lambda_main'],
            'loss_mse': align_loss_dict['mse_loss'] * self.config['lambda_q'],
            'loss_margin': align_loss_dict['margin_loss'] * self.config['align_margin_loss_kwargs'][
                'margin_lambda'],
            'loss_norm': align_loss_dict['norm_loss'] * self.config['align_margin_loss_kwargs']['norm_lambda'],
            'student_MI': student_terms['MI'] * self.config['alpha_student'],
            'student_TC': student_terms['TC'] * self.config['beta_student'],
            'student_TC_G': student_terms['TC_G'] * self.config['g_beta_student'],
            'student_KL': student_terms['KL'] * self.config['gamma_student'],
            'exercise_MI': exercise_terms['MI'] * self.config['alpha_exercise'],
            'exercise_TC': exercise_terms['TC'] * self.config['beta_exercise'],
            'exercise_TC_G': exercise_terms['TC_G'] * self.config['g_beta_exercise'],
            'exercise_KL': exercise_terms['KL'] * self.config['gamma_exercise'],
            'exercise_MI_diff': exercise_terms_diff['MI'] * self.config['alpha_exercise'],
            'exercise_TC_diff': exercise_terms_diff['TC'] * self.config['beta_exercise'],
            'exercise_TC_G_diff': exercise_terms_diff['TC_G'] * self.config['g_beta_exercise'],
            'exercise_KL_diff': exercise_terms_diff['KL'] * self.config['gamma_exercise'],
        }

    def get_tcvae_terms(self, z, params, dist, dataset_size):
        batch_size, latent_dim = z.shape

        if isinstance(dist, NormalDistUtil):
            mu, logvar = params
            zero = torch.FloatTensor([0.0]).to(z.device)
            logpz = dist.log_density(X=z, MU=zero, LOGVAR=zero).sum(dim=1)
            logqz_condx = dist.log_density(X=z, MU=mu, LOGVAR=logvar).sum(dim=1)
            _logqz = dist.log_density(
                z.reshape(batch_size, 1, latent_dim),
                mu.reshape(1, batch_size, latent_dim),
                logvar.reshape(1, batch_size, latent_dim)
            )  # _logqz的第(i,j,k)个元素, P(z(n_i)_k|n_j)
        elif isinstance(dist, BernoulliUtil):
            logpz = dist.log_density(z, params=None).sum(dim=1)
            logqz_condx = dist.log_density(z, params=params).sum(dim=1)
            # _logqz = torch.stack([dist.log_density(z, params[i,:]) for i in range(batch_size)],dim=1)
            _logqz = dist.log_density(z.reshape(batch_size, 1, latent_dim),
                                      params=params.reshape(1, batch_size, latent_dim), is_check=False)
        else:
            raise ValueError("unknown base class of dist")

        if self.config['sampling_type'] == 'mws':
            # minibatch weighted sampling
            logqz_prodmarginals = (
                    torch.logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (torch.logsumexp(_logqz.sum(dim=2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
            logqz_group_list = []
            if hasattr(self, 'dict_cpt_relation'):
                for gid, group_idx in self.dict_cpt_relation.items():
                    logqz_group_list.append(
                        (torch.logsumexp(_logqz[:, :, group_idx].sum(dim=2), dim=1, keepdim=False) - math.log(
                            batch_size * dataset_size))
                    )
                logqz_group = torch.vstack(logqz_group_list).T.sum(dim=1)
        elif self.config['sampling_type'] == 'mss':
            logiw_mat = self._log_importance_weight_matrix(z.shape[0], dataset_size).to(z.device)
            logqz = torch.logsumexp(
                logiw_mat + _logqz.sum(dim=-1), dim=-1
            )  # MMS [B]
            logqz_prodmarginals = (
                torch.logsumexp(
                    logiw_mat.reshape(z.shape[0], z.shape[0], -1) + _logqz,
                    dim=1,
                )
            ).sum(
                dim=-1
            )
            logqz_group_list = []
            if hasattr(self, 'dict_cpt_relation'):
                for gid, group_idx in self.dict_cpt_relation.items():
                    logqz_group_list.append(
                        (
                            torch.logsumexp(
                                logiw_mat.reshape(z.shape[0], z.shape[0], -1) + _logqz[:, :, group_idx], dim=1,
                            )).sum(dim=-1)
                    )
                logqz_group = torch.vstack(logqz_group_list).T.sum(dim=1)

        else:
            raise ValueError("Unknown Sampling Type")

        IndexCodeMI = logqz_condx - logqz
        TC = logqz - logqz_prodmarginals
        TC_G = (logqz - logqz_group).mean() if hasattr(self, 'dict_cpt_relation') else torch.FloatTensor([0.0]).to(
            z.device)
        DW_KL = logqz_prodmarginals - logpz
        return {
            'MI': IndexCodeMI.mean(),
            'TC': TC.mean(),
            'TC_G': TC_G,
            'KL': DW_KL.mean()
        }

    def get_main_loss(self, **kwargs):
        student_id = kwargs['student_id']
        exercise_id = kwargs['exercise_id']
        r = kwargs['r']
        return self.forward(student_id, exercise_id, r)

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)

    def diagnose(self):
        student_mix = self.dcdModules.EncoderStudent(self.interact_mat)
        student_emb, _ = torch.chunk(student_mix, 2, dim=-1)
        # student_emb = self.student_dist.sample(student_mu, student_logvar)
        return student_emb


    def get_attribute(self, attribute_name):
        if attribute_name == 'mastery':
            return self.diagnose().detach().cpu().numpy()
        elif attribute_name == 'diff':
            return self.inter_func.transform(self.extractor["diff"],
                                             self.extractor["knowledge"]).detach().cpu().numpy()
        elif attribute_name == 'knowledge':
            return self.extractor["knowledge"].detach().cpu().numpy()
        else:
            return None
