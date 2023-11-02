import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from tqdm import tqdm

from .._base import _InteractionFunction
from ..datahub import DataHub
from ._util import none_neg_clipper


class NCD_IF(_InteractionFunction, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, hidden_dims: list, dropout, device, dtype):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        self.__student_emb = nn.Embedding(self.student_num, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.__diff_emb = nn.Embedding(self.exercise_num, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)

        self.__emb_map = {
            "mastery": self.__student_emb,
            "diff": self.__diff_emb,
            "disc": self.__disc_emb
        }

        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.knowledge_num, hidden_dim, dtype=self.dtype),
                        'activation0': nn.Tanh()
                    }
                )
            else:
                layers.update(
                    {
                        'dropout{}'.format(idx): nn.Dropout(p=self.dropout),
                        'linear{}'.format(idx): nn.Linear(self.hidden_dims[idx-1], hidden_dim, dtype=self.dtype),
                        'activation{}'.format(idx): nn.Tanh()
                    }
                )
        layers.update(
            {
                'dropout{}'.format(len(self.hidden_dims)): nn.Dropout(p=self.dropout),
                'linear{}'.format(len(self.hidden_dims)): nn.Linear(
                    self.hidden_dims[len(self.hidden_dims) - 1], 1, dtype=self.dtype
                ),
                'activation{}'.format(len(self.hidden_dims)): nn.Sigmoid()
            }
        )

        self.mlp = nn.Sequential(layers).to(self.device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_id, exercise_id, q_mask):
        student_emb = torch.sigmoid(self.__student_emb(student_id))
        diff_emb = torch.sigmoid(self.__diff_emb(exercise_id))
        disc_emb = torch.sigmoid(self.__disc_emb(exercise_id))
        input_x = disc_emb * (student_emb - diff_emb) * q_mask
        return self.mlp(input_x).view(-1)

    def fit(self, datahub: DataHub, set_type, **kwargs):
        dataloader = datahub.to_dataloader(
            batch_size=kwargs["batch_size"],
            dtype=self.dtype,
            set_type=set_type,
            label=True
        )
        loss_func = kwargs["loss_func"]
        optimizer = kwargs["optimizer"]
        epoch_losses = []
        self.train()
        for batch_data in tqdm(dataloader, "Training"):
            student_id, exercise_id, knowledge, r = batch_data
            student_id: torch.Tensor = student_id.to(self.device)
            exercise_id: torch.Tensor = exercise_id.to(self.device)
            knowledge: torch.Tensor = knowledge.to(self.device)
            r: torch.Tensor = r.to(self.device)
            pred_r = self.forward(student_id, exercise_id, knowledge)
            loss = loss_func(pred_r, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # clipper for positive
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    layer.apply(none_neg_clipper)

            epoch_losses.append(loss.mean().item())
        print("Average loss: {}".format(float(np.mean(epoch_losses))))

    def compute(self, datahub: DataHub, set_type, **kwargs):
        dataloader = datahub.to_dataloader(
            batch_size=kwargs["batch_size"],
            dtype=self.dtype,
            set_type=set_type,
            label=False
        )
        self.eval()
        pred = []
        for batch_data in tqdm(dataloader, "Evaluating"):
            student_id, exercise_id, knowledge = batch_data
            student_id: torch.Tensor = student_id.to(self.device)
            exercise_id: torch.Tensor = exercise_id.to(self.device)
            knowledge = knowledge.to(self.device)
            pred_r = self.forward(student_id, exercise_id, knowledge)
            pred.extend(pred_r.detach().cpu().tolist())
        return pred


    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can detach {} from embeddings.".format(self.__emb_map.keys()))
        return torch.sigmoid(self.__emb_map[item].weight.detach().cpu()).numpy()