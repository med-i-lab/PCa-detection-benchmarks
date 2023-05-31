
from projects.IJCARS_2023.base import BaseExperiment
import torch 

class FinetuningExperiment(BaseExperiment):
    def create_model(self):
        from src.modeling.registry import create_model
        feature_extractor = create_model(
            self.args.model_name,   
        ) 
        feature_extractor.load_state_dict(torch.load(self.args.weights_path))
        linear_layer = torch.nn.Linear(512, 2) 

        self.feature_extractor = feature_extractor
        self.linear_layer = linear_layer
        return torch.nn.Sequential(feature_extractor, linear_layer).cuda()
   
    def create_optimizer(self, model):
        opt = torch.optim.Adam(
            [
                {
                    "params": self.feature_extractor.parameters(),
                    "lr": self.args.lr_finetune,
                }, 
                {
                    "params": self.linear_layer.parameters(),
                    "lr": self.args.lr,
                },
            ]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.args.num_epochs)
        return opt, sched
    