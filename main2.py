import numpy as np
import torch
import wandb

from pprint import pprint
from inscd import listener
from inscd.datahub import DataHub
from inscd.models.static.neural import NCDM
from inscd.models.static.graph import ULCDF
from inscd.models.static.neural import KANCD
from inscd.models.static.classic import MIRT

# wandb.init(
#     project="test inscd"
# )

listener.update(print)
seed = 5
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

datahub = DataHub("datasets/Assist17")
datahub.random_split(source="total", to=["train", "test"], seed=seed)
print("Number of response logs {}".format(len(datahub)))

# kancd = KANCD(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
# kancd.build(20, device='cuda:0')
# kancd.train(datahub, "train", "test", valid_metrics=['auc', 'ap', 'doa'], batch_size=256, weight_decay=0)

ulcdf = ULCDF(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
ulcdf.build(latent_dim=32, device='cuda:1', predictor_type='dp-linear', gcn_layers=3, dtype=torch.float64)
ulcdf.train(datahub, "train", "test", valid_metrics=['auc', 'ap', 'doa'], batch_size=1024, lr=6e-3, weight_decay=0, epoch=20)

# ncdm = NCDM(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
# ncdm.build()
# ncdm.train(datahub, "train", "test", valid_metrics=['auc', 'ap', 'doa'], batch_size=32)
# test_results = ncdm.score(datahub, "test", metrics=['auc', 'ap', 'doa'])
# pprint(test_results)

# ncdm = NCDM(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
# ncdm.build()
# ncdm.train(datahub, "train", "test", valid_metrics=['auc', 'ap', 'doa'], batch_size=32)

# mirt = MIRT(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
# mirt.build(latent_dim=8, device='cuda:0', if_type='sum')
# mirt.train(datahub, "train", "test", valid_metrics=['auc', 'ap'], batch_size=256, lr=0.01, weight_decay=0, epoch=20)
