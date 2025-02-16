from .basemodel import Basemodel
from .models import PINN_Net, CustomPINN
from src.Training.trainer import Trainer
from src.Operators.Bound_Op import BoundaryCondition, BoundaryLocation, BoundaryType    
from src import pdeOperator , OperatorConfig
import torch
import typing as tp
#from src.utils.PDE1 import load_data


class PINNModel(Basemodel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

        self.model = PINN_Net(1, 1, self.param['N_hidden'], self.param['N_layers'])

        self.boundary_conditions = [
                BoundaryCondition(
                    type=BoundaryType.DIRICHLET,
                    location=BoundaryLocation.X_MAX,    
                    value= self.load_data()['bound_left'] ,
                    weight= torch.tensor(1.0),#torch.tensor(self.param['weight']),
                    trainable=self.param['TF'],
                    weight_function= lambda x: torch.exp(-x),

                ),
                
                BoundaryCondition(
                    type=BoundaryType.DIRICHLET,
                    location=BoundaryLocation.X_MIN,
                    value= self.load_data()['bound_right'],
                    weight= torch.tensor(self.param['weight']),
                    trainable=self.param['TF'],
                    weight_function = lambda x: torch.exp(-x),
                )
            ]

                


    def fit(self ,x): 
        x = torch.sort(x)[0]
        x.requires_grad = True  

        trainer = Trainer(
            [x], self.boundary_conditions, self.pde_configurations, model=self.model , watch=self.param['wandb_logs'] , name= self.param['model_name'], lr=self.param['lr'])

        mse, loss = trainer.train(
            epochs=self.param['epochs'], rate=self.pde_configurations.update_rate, loss_function=self.pde_configurations.pde_loss
        )

        self.model = trainer.model

        return mse, loss
    

    def predict(self, x):
        return self.model(x)
    

    def save(self, path):
        torch.save(self.model.state_dict(), path)