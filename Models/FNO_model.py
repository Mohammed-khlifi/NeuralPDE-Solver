from .NO_basemodel import NO_basemodel
import torch
from neuraloperator.neuralop.models import FNO
from neuraloperator.neuralop import Trainer


class FNO_model(NO_basemodel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        self.model = FNO(n_modes=(16, 16),
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            projection_channel_ratio=2)
        
        self.learning_rate = self.param['lr']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.1)
        self.trainer = Trainer(model= self.model, n_epochs= 100,
                  device='cpu',
                  wandb_log=True,
                  eval_interval=3,
                  use_distributed=False,
                  verbose=True) 

