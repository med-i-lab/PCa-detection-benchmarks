from projects.IJCARS_2023.utils import *
from projects.IJCARS_2023.base import BaseExperiment
from src.modeling.loss.isomax import IsoMaxPlusLossFirstPart, IsoMaxPlusLossSecondPart
import torch
from torch.nn import functional as F


class IsoMaxModel(torch.nn.Module):
    def __init__(self, feature_extractor, num_features, num_classes, temperature=1.0):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.isomax_loss_first_part = IsoMaxPlusLossFirstPart(
            num_features=num_features, num_classes=num_classes, temperature=temperature
        )

    def forward(self, x, return_mds=False):
        x = self.feature_extractor(x)
        logits, mds = self.isomax_loss_first_part(x, return_mds=True)
        if return_mds:
            return logits, mds
        return logits


class IsoMax(BaseExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.loss_fn = IsoMaxPlusLossSecondPart(
            entropic_scale=self.args.entropic_scale,
        ).to(self.args.device)

    def train_step(self, model, batch, optimizer, epoch):
        optimizer.zero_grad()
        x, y, metadata = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)
        y_hat, mds = model(x, return_mds=True)
        prob = F.softmax(y_hat, dim=1)
        pred = prob.argmax(dim=1)
        confidence = -mds
        loss = F.cross_entropy(y_hat, y)
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
        x, y, metadata = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)
        y_hat, mds = model(x, return_mds=True)
        prob = F.softmax(y_hat, dim=1)
        pred = prob.argmax(dim=1)
        confidence = -mds
        return {
            "loss": F.cross_entropy(y_hat, y, reduction="none"),
            "y": y,
            "prob": prob,
            "confidence": confidence,
            **metadata,
        }

    def create_model(self):
        from src.modeling.registry import resnet10_feature_extractor

        model = resnet10_feature_extractor()
        model = IsoMaxModel(model, num_features=512, num_classes=2)

        return model.cuda()
