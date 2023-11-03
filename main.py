import wandb

from pprint import pprint
from inteledu import listener
from inteledu.datahub import DataHub
from inteledu.models.cross.neural import NCDM

wandb.init(
    project="test inteledu"
)

listener.update(wandb.log)

datahub = DataHub("datasets/Math2")
datahub.random_split()
datahub.random_split(source="valid", to=["valid", "test"])
print("Number of response logs {}".format(len(datahub)))

ncdm = NCDM(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
ncdm.build()
ncdm.train(datahub, "train", "valid")
test_results = ncdm.score(datahub, "test", metrics=["acc", "doa"])
pprint(test_results)

