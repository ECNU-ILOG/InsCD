import os
import torch
import numbers
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import scipy.sparse as sp

from collections import OrderedDict
from sklearn.cluster import KMeans
from ..._base import _CognitiveDiagnosisModel
from ...interfunc import get_interaction_function
from ...extractor import HyperCDM_EX
import torch.nn.functional as f
from joblib import Parallel, delayed


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.hidden_dims = hidden_dims
        self.hidden_dims.append(latent_dim)
        self.dims_list = (hidden_dims + hidden_dims[:-1][::-1])  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == self.latent_dim

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.input_dim, hidden_dim),
                        'activation0': nn.ReLU()
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            self.hidden_dims[idx-1], hidden_dim),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            self.hidden_dims[idx])
                    }
                )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, self.output_dim),
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, tmp_hidden_dims[idx+1]),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            tmp_hidden_dims[idx+1])
                    }
                )
        self.decoder = nn.Sequential(layers)

    def __repr__(self):
        repr_str = '[Structure]: {}-'.format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
            repr_str += '{}-'.format(dim)
        repr_str += str(self.output_dim) + '\n'
        repr_str += '[n_layers]: {}'.format(self.n_layers) + '\n'
        repr_str += '[n_clusters]: {}'.format(self.n_clusters) + '\n'
        repr_str += '[input_dims]: {}'.format(self.input_dim)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        output = self.encoder(X)
        if latent:
            return output
        return self.decoder(output)


def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat


class BatchKMeans(object):
    def __init__(self, latent_dim, n_clusters, n_jobs):
        self.n_features = latent_dim
        self.n_clusters = n_clusters
        self.clusters = np.zeros((self.n_clusters, self.n_features))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = n_jobs

    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i])
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)

        return dis_mat

    def init_cluster(self, X):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters, n_init=20)
        model.fit(X)
        self.clusters = model.cluster_centers_  # copy clusters

    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] +
                               eta * X[i])
            self.clusters[cluster_idx] = updated_cluster

    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        dis_mat = self._compute_dist(X)
        return np.argmin(dis_mat, axis=1)

    def assign_group(self, X, belong):
        dis_mat = self._compute_dist(X)
        return np.argsort(dis_mat, axis=1)[:, :belong]


class DeepClusteringNet(nn.Module):
    # def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters, beta=1, lamda=1,
    #              pretrained=True, lr=0.0001, n_jobs=-1, device="cuda", datahub_name='Math1'):
    def __init__(self, config, device='cuda', n_clusters=64):
        super(DeepClusteringNet, self).__init__()
        input_dim, hidden_dims, latent_dim = int(config["exercise_num"]), config['ae_hidden_dims'],config['ae_latent_dim']
        self.beta = config['beta']  # coefficient of the clustering term
        self.lamda = config['lamda']  # coefficient of the reconstruction term
        self.device = device
        self.pretrained = config['pretrained']
        self.n_clusters = n_clusters
        self.config = config

        # Validation check
        if not self.beta > 0:
            msg = 'beta should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.beta))

        if not self.lamda > 0:
            msg = 'lambda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lamda))

        if len(hidden_dims) == 0:
            raise ValueError('No hidden layer specified.')

        self.kmeans = BatchKMeans(latent_dim, n_clusters, self.config['n_jobs'])
        self.autoencoder = AutoEncoder(input_dim, hidden_dims, latent_dim, n_clusters).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.config['cluster_lr'],
                                          weight_decay=5e-4)

    def _loss(self, X, cluster_id):
        batch_size = X.size()[0]
        rec_X = self.autoencoder(X)
        latent_X = self.autoencoder(X, latent=True)

        # Reconstruction error
        rec_loss = self.lamda * self.criterion(X, rec_X)

        # Regularization term on clustering
        dist_loss = torch.tensor(0.).to(self.device)
        clusters = torch.FloatTensor(self.kmeans.clusters).to(self.device)
        for i in range(batch_size):
            diff_vec = latent_X[i] - clusters[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1))
            dist_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss)

        return (rec_loss + dist_loss,
                rec_loss.detach().cpu().numpy(),
                dist_loss.detach().cpu().numpy())

    def pretrain(self, train_loader, epoch=100):
        if not self.pretrained:
            return
        if not isinstance(epoch, numbers.Integral):
            msg = '`epoch` should be an integer but got value = {}'
            raise ValueError(msg.format(epoch))

        rec_loss_list = []

        self.train()
        for e in tqdm(range(epoch), "gain feature"):
            for data in train_loader:
                batch_size = data.size()[0]
                data = data.to(self.device).view(batch_size, -1)
                rec_X = self.autoencoder(data)
                loss = self.criterion(data, rec_X)

                rec_loss_list.append(loss.detach().cpu().numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.eval()

        # Initialize clusters in self.kmeans after pre-training
        batch_X = []
        for data in train_loader:
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.autoencoder(data, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        self.kmeans.init_cluster(batch_X)

        return rec_loss_list

    def fit(self, train_loader, epoch=50):
        for e in tqdm(range(epoch), "clustering"):
            self.train()
            for data in train_loader:
                batch_size = data.size()[0]
                data = data.view(batch_size, -1).to(self.device)

                # Get the latent features
                with torch.no_grad():
                    latent_X = self.autoencoder(data, latent=True)
                    latent_X = latent_X.cpu().numpy()

                # [Step-1] Update the assignment results
                cluster_id = self.kmeans.update_assign(latent_X)

                # [Step-2] Update clusters in bath Kmeans
                elem_count = np.bincount(cluster_id,
                                         minlength=self.n_clusters)
                for k in range(self.n_clusters):
                    # avoid empty slicing
                    if elem_count[k] == 0:
                        continue
                    self.kmeans.update_cluster(latent_X[cluster_id == k], k)

                # [Step-3] Update the network parameters
                loss, rec_loss, dist_loss = self._loss(data, cluster_id)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def gain_clusters(self, train_loader, belong):
            # Create filename based on datahub_name
        datahub_name = self.config['datahub_name']
        filename = os.path.join('ckpt', f'cluster_groups_{datahub_name}.npy')
        

        clusters = []
        for data in train_loader:
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)

            with torch.no_grad():
                latent_X = self.autoencoder(data, latent=True)
                latent_X = latent_X.cpu().numpy()

            cluster_id = self.kmeans.assign_group(latent_X, belong)
            clusters.append(cluster_id)
        group_ids = np.vstack(clusters)
            # Save the computed groups
        os.makedirs('ckpt', exist_ok=True)  # Create directory if it doesn't exist
        np.save(filename, group_ids)
        return group_ids


class Hypergraph:
    def __init__(self, H: np.ndarray):
        self.H = H
        # avoid zero
        self.Dv = np.count_nonzero(H, axis=1) + 1
        self.De = np.count_nonzero(H, axis=0) + 1

    def to_tensor_nadj(self):
        coo = sp.coo_matrix(self.H @ np.diag(1 / self.De) @ self.H.T @ np.diag(1 / self.Dv))
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()


class HyperCDM(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        Hypergraph Cognitive Diagnosis (HyperCDM)
        Junhao Shen et al. Capturing Homogeneous Influence among Students: Hypergraph Cognitive Diagnosis for Intelligent Education Systems. KDD'24.
        """
        super().__init__(config=config)
        self.extractor = HyperCDM_EX(config)
        inter_func_class, require_transfer = get_interaction_function(config.get('if_type', 'NCD_IF'))
        self.inter_func = inter_func_class(config)
        self.config['require_transfer'] = require_transfer
        self.build_graph()
    
    def build_graph(self):
        graph_dict = {
            'student_adj': self.hyper_construct("student").to_tensor_nadj().float(),
        'exercise_adj': self.hyper_construct("exercise").to_tensor_nadj().float(),
        'knowledge_adj':self.hyper_construct("knowledge").to_tensor_nadj().float()}
        self.extractor.get_graph_dict(graph_dict)

    
    def hyper_construct(self, choice="student"):
        # Only use train set to avoid leakage
        if choice == "student":
            print("Construct student hypergraph")
            """
            strategy: deep clustering ==> hypergraph
            """
            X = torch.tensor(self.get_r_matrix(choice="train")).float()
            n_clusters = int(int(self.config["student_num"]) * 0.02)
            student_response_loader = torch.utils.data.DataLoader(dataset=X, batch_size=256, shuffle=False)

            datahub_name = self.config['datahub_name']
            filename = os.path.join('ckpt', f'cluster_groups_{datahub_name}.npy')
        
        # Check if file exists
            if os.path.exists(filename):
            # Load existing groups
                groups = np.load(filename)
            else:
                clf = DeepClusteringNet(self.config,
                                    n_clusters=n_clusters)
                clf.pretrain(student_response_loader)
                clf.fit(student_response_loader)
                groups = clf.gain_clusters(student_response_loader, n_clusters // 2)
            H = np.zeros((int(self.config["student_num"]), n_clusters))
            for i in range(H.shape[0]):
                H[i, groups[i]] = 1
            H = H[:, np.count_nonzero(H, axis=0) >= 2]  # remove empty edge
        elif choice == "exercise":
            print("Construct exercise hypergraph")
            H = self.config['datahub'].q_matrix.copy()
            H = H[:, np.count_nonzero(H, axis=0) >= 2]  # remove empty edge
        elif choice == "knowledge":
            print("Construct knowledge concept hypergraph")
            H = self.config['datahub'].q_matrix.T.copy()
            H = H[:, np.count_nonzero(H, axis=0) >= 2]  # remove empty edge
        else:
            raise ValueError("Only \"student\", \"exercise\" and \"knowledge\" are capable for parameter choice")
        return Hypergraph(H)

    def get_r_matrix(self, choice="train"):
        r_matrix = -1 * np.ones(shape=(int(self.config["student_num"]), int(self.config["exercise_num"])))
        if choice == "train":
            for line in self.config['datahub']['train']:
                student_id = int(line[0])
                exercise_id = int(line[1])
                score = int(line[2])
                r_matrix[student_id, exercise_id] = int(score)
        elif choice == "test":
            for line in self.config['datahub']['test']:
                student_id = int(line[0])
                exercise_id = int(line[1])
                score = line[2]
                r_matrix[student_id, exercise_id] = int(score)
        elif choice == "total":
            for line in self.config['datahub']['train']:
                student_id = int(line[0])
                exercise_id = int(line[1])
                score = line[2]
                r_matrix[student_id, exercise_id] = int(score)
            for line in self.config['datahub']['test']:
                student_id = int(line[0])
                exercise_id = int(line[1])
                score = line[2]
                r_matrix[student_id, exercise_id] = int(score)
        else:
            raise ValueError("Only \"train\" and \"test\" are capable for parameter choice")
        return r_matrix
    
    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])