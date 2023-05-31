from projects.IJCARS_2023.utils import *
from .base import BaseExperiment
import torch


class SNGP(BaseExperiment):
    def train_epoch(self, model, train_loader, optimizer, epoch):
        model.reset_precision_matrix()
        return super().train_epoch(model, train_loader, optimizer, epoch)

    def create_model(
        self,
    ):
        from src.modeling.GP_approx_models import Laplace
        from src.modeling.spectral_resnets import spectral_resnet10

        feature_extractor = spectral_resnet10(in_channels=1, num_classes=2)
        feature_extractor.fc = torch.nn.Identity()

        len_data = len(self.train_ds)
        model = Laplace(
            feature_extractor,
            num_deep_features=512,
            num_gp_features=128,
            num_data=len_data,
            mean_field_factor=25,
            train_batch_size=self.args.batch_size,
            feature_scale=self.args.feature_scale,
            ridge_penalty=self.args.ridge_penalty,
        )
        return model.cuda()
