import numpy as np
import torch
import wandb

from pprint import pprint
from inscd import listener
from inscd.datahub import DataHub
from inscd.models.static.neural import NCDM
from inscd.models.static.graph import ULCDF
# wandb.init(
#     project="test inscd"
# )

listener.update(print)
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

datahub = DataHub("datasets/Math1")
datahub.random_split(source="total", to=["train", "test"], seed=seed)
print("Number of response logs {}".format(len(datahub)))

ulcdf = ULCDF(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
ulcdf.build(latent_dim=256, device='cuda:0', predictor_type='dp-linear')
ulcdf.train(datahub, "train", "test", valid_metrics=['auc', 'ap', 'doa'], batch_size=32)
# ncdm = NCDM(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
# ncdm.build()
# ncdm.train(datahub, "train", "test", valid_metrics=['auc', 'ap', 'doa'], batch_size=32)
# test_results = ncdm.score(datahub, "test", metrics=['auc', 'ap', 'doa'])
# pprint(test_results)
