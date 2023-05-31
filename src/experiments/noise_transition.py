from projects.IJCARS_2023.base import BaseExperiment
import torch
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class AddNoiseTransition(BaseExperiment):
    def create_model(self):
        from src.modeling.noise_transition_matrix import NoiseTransitionMatrix

        model = super().create_model()
        self.noise_transition_matrix = NoiseTransitionMatrix(2, initial_temperature=1)
        return torch.nn.Sequential(model, self.noise_transition_matrix).cuda()

    def _log_noise_transition_matrix(self):
        noise_transition_matrix = (
            self.noise_transition_matrix.noise_transition_matrix.softmax(1)
        )
        noise_transition_matrix = noise_transition_matrix.detach().cpu().numpy()
        import numpy as np
        import seaborn as sns

        plt.figure(figsize=(2, 2))
        sns.heatmap(
            noise_transition_matrix,
            annot=True,
            cmap="Blues",
            cbar=False,
            square=True,
            fmt=".4f",
        )

        wandb.log(
            {"noise_transition_matrix": wandb.Image(plt.gcf()), "epoch": self.epoch}
        )

    def train_epoch(self, model, train_loader, optimizer, epoch):
        out = super().train_epoch(model, train_loader, optimizer, epoch)
        self._log_noise_transition_matrix()
        return out
