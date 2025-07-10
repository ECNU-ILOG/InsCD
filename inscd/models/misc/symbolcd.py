import torch

# from ... import listener, ruler
from ... import listener
from ..._base import _CognitiveDiagnosisModel
from ._symbolcd import GeneticInteractionFunc, Parameter


class SymbolCD(_CognitiveDiagnosisModel):
    def __init__(self, config):
        """
        Description:
        Symbolic Cognitive Diagnosis Model (SymbolCD)
        Junhao Shen et al. Symbolic Cognitive Diagnosis via Hybrid Optimization for Intelligent Education Systems. AAAI'24.
        """
        super().__init__(config=config)
        
        self.extractor = Parameter(config)
        
        self.inter_func = GeneticInteractionFunc()


    def predict(self, datahub, set_type: str, batch_size=256, **kwargs):
        return self.extractor.predict(datahub, set_type, batch_size, **kwargs)

    def symbolic_train_parameters(self, datahub, set_type="train",
              valid_set_type=None, valid_metrics=None, epoch=10, para_epoch=100, lr=0.002,
              population_size=200, ngen=10, cxpb=0.5, mutpb=0.1, batch_size=256, **kwargs):
        for epoch_i in range(0, epoch):
            print("[Epoch {}]".format(epoch_i + 1))
            self.extractor.train(datahub, set_type, epoch=para_epoch, lr=lr, init=(epoch == 0), batch_size=batch_size)
            print(f"The {epoch_i + 1}-th epoch extractor optimization complete")
            arguments = self.extractor.unpack()
            self.inter_func.update(*arguments)
            self.inter_func.train(datahub, set_type, population_size, ngen, cxpb, mutpb, batch_size)
            print(f"The {epoch_i + 1}-th epoch interaction function complete")
            self.extractor.update(self.inter_func.function(), str(self.inter_func))
        if valid_set_type is not None:
            self.score(datahub, valid_set_type, valid_metrics, batch_size=batch_size, **kwargs)

    # @listener
    # def score(self, datahub, set_type, metrics: list, batch_size=256, **kwargs) -> dict:
    #     if metrics is None:
    #         metrics = ["acc", "auc", "f1", "doa", 'ap']
    #     if self.inter_func is Ellipsis or self.extractor is Ellipsis:
    #         raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
    #     pred_r = self.extractor.predict(datahub, set_type, batch_size, **kwargs)
    #     return ruler(self, datahub, set_type, pred_r, metrics)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.extractor.net.student_emb.weight


