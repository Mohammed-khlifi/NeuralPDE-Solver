from .basemodel import Basemodel
from .NO_basemodel import NO_basemodel
from .models import PINN_Net, CustomPINN
from src.Operators.Bound_Op import BoundaryCondition, BoundaryLocation, BoundaryType    
from src import pdeOperator , OperatorConfig
import torch
import typing as tp
from src import FNO
from src.neuraloperator.neuralop import Trainer
from src.neuraloperator.neuralop import LpLoss, H1Loss
import torch.nn.functional as F

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

    """def fit(self, train_loader , test_loaders , epochs=100):


        train_loss = H1Loss(d=2)
        l2loss = LpLoss(d=2 ,p=2)
        val_loss = {'h1': train_loss, 'l2': l2loss}

        self.trainer.train(train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=self.optimizer,
            scheduler= self.scheduler,
            regularizer=False,
            training_loss= train_loss,
            eval_losses=val_loss,)"""