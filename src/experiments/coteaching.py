from projects.IJCARS_2023.base import BaseExperiment
import torch
import wandb
import numpy as np
from src.modeling.loss.coteaching import coteaching_loss
from torch.nn import functional as F


class CoTeaching(BaseExperiment):
    def create_model(self):
        model1 = super().create_model()
        model2 = super().create_model()

        return torch.nn.ModuleList([model1, model2]).cuda()

    def get_forget_rate(self):
        x = self.epoch
        x1 = 0.5 * self.args.num_epochs
        if self.args.get("final_forget_rate", None) is not None:
            y1 = self.args.final_forget_rate
        else:
            y1 = 0.5
        a = y1 / x1

        return min(a * x, y1)

    def train_step(self, model, batch, optimizer, epoch):
        optimizer.zero_grad()
        x, y, metadata = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)

        y_hat = model[0](x)
        prob = F.softmax(y_hat, dim=1)
        pred = prob.argmax(dim=1)
        confidence = prob.max(dim=1).values
        loss = F.cross_entropy(y_hat, y, reduction="none")

        y_hat2 = model[1](x)
        prob2 = F.softmax(y_hat2, dim=1)
        pred2 = prob2.argmax(dim=1)
        confidence2 = prob2.max(dim=1).values
        loss2 = F.cross_entropy(y_hat2, y, reduction="none")

        loss = coteaching_loss(loss, loss2, self.get_forget_rate(), self.args.device)
        loss.backward()
        optimizer.step()

        return {
            "loss": F.cross_entropy(y_hat, y, reduction="none"),
            "y": y,
            "prob": prob,
            "confidence": confidence,
            **metadata,
        }

    def eval_step(self, model, batch):
        return super().eval_step(model[0], batch)

    def train_epoch(self, model, train_loader, optimizer, epoch):
        out = super().train_epoch(model, train_loader, optimizer, epoch)
        wandb.log(
            {"forget_rate": self.get_forget_rate(), "epoch": self.epoch},
        )
        return out


class SNGPCoTeaching(CoTeaching):
    def create_model(self):
        from projects.IJCARS_2023.sngp import SNGP

        model1 = SNGP.create_model(self)
        model2 = SNGP.create_model(self)
        return torch.nn.ModuleList([model1, model2]).cuda()

    def train_epoch(self, model, train_loader, optimizer, epoch):
        model[0].reset_precision_matrix()
        model[1].reset_precision_matrix()
        return super().train_epoch(model, train_loader, optimizer, epoch)
