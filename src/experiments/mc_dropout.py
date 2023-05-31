from projects.IJCARS_2023.utils import *
from projects.IJCARS_2023.base import BaseExperiment
from torch.nn import Module
from torch import nn
from torch.nn import functional as F


class MCDropoutModel(Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x, n=1):
        return torch.stack([self.module(x) for _ in range(n)], dim=0).mean(0)

    def _turn_on_dropout(self, module):
        if isinstance(module, nn.Dropout):
            module.train()

    def eval(self):
        self.module.eval()
        self.apply(lambda module: self._turn_on_dropout(module))


class MCDropout(BaseExperiment):
    def eval_step(self, model, batch):
        x, y, metadata = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)
        y_hat = model.mc_dropout(x, n=self.args.n_passes)
        prob = F.softmax(y_hat, dim=1)
        confidence = prob.max(dim=1).values
        return {
            "loss": F.cross_entropy(y_hat, y, reduction="none"),
            "y": y,
            "prob": prob,
            "confidence": confidence,
            **metadata,
        }

    def create_model(self):
        model = super().create_model()
        return MCDropoutModel(model)
