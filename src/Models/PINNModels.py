from .basemodel import Basemodel
from .models import PINN_Net, CustomPINN
from src.Training.trainer import Trainer
from src.Operators.Bound_Op import BoundaryCondition, BoundaryLocation, BoundaryType    
from src import pdeOperator , OperatorConfig
import torch
import typing as tp



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
                    weight= torch.tensor(1.0),
                    trainable=self.param['TF'],
                    weight_function = lambda x: torch.exp(-x),
                )
            ]


class PINNModel_2D(Basemodel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

        self.model =  PINN_Net(2, 1, self.param['N_hidden'], self.param['N_layers'])
        weight= torch.tensor(self.param['weight'])
        self.boundary_conditions = [
                BoundaryCondition(
                        type=BoundaryType.DIRICHLET,
                        location=BoundaryLocation.Y_MAX,
                        value= self.load_data()['bound_up'] ,#lambda x,y : torch.exp(-10*(y[0,:]**2 + 1)).squeeze(),
                        weight= weight,
                        trainable=self.param['TF'],
                        weight_function= lambda x : torch.exp(x),
                        ),
                BoundaryCondition(
                        type=BoundaryType.DIRICHLET,
                        location=BoundaryLocation.Y_MIN,
                        value= self.load_data()['bound_down'] ,#lambda x,y : torch.exp(-10*(y[0,:]**2 + 1)).squeeze(),
                        weight= weight,
                        trainable=self.param['TF'],
                        weight_function= lambda x : torch.exp(x),


                        ),
                BoundaryCondition(
                        type=BoundaryType.DIRICHLET,
                        location=BoundaryLocation.X_MIN,
                        value= self.load_data()['bound_right'] ,#lambda x,y : torch.exp(-10*(1 + x[:,0]**2)).squeeze(),
                        weight= weight,
                        trainable= self.param['TF'],
                        weight_function= lambda x : torch.exp(x),
                        ),
                BoundaryCondition(
                        type=BoundaryType.DIRICHLET,
                        location=BoundaryLocation.X_MAX,
                        value= self.load_data()['bound_left'] ,#lambda x,y : torch.exp(-10*(1 + x[:,0]**2)).squeeze(),
                        weight= weight,
                        trainable= self.param['TF'],
                        weight_function= lambda x : torch.exp(x),
                        )
        ]



class PINNModel_3D(Basemodel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

        self.model = PINN_Net(3, 1, self.param['N_hidden'], self.param['N_layers'])
        weight= torch.tensor(self.param['weight'])
        self.boundary_conditions = [
            BoundaryCondition(
                    type=BoundaryType.DIRICHLET,
                    location=BoundaryLocation.Z_MIN,
                    value= self.load_data()['bound_in'],#lambda x,y,z : self.u_exact(x , y , z)[0, :, :],
                    weight= weight,
                    trainable=self.param['TF'],
                    weight_function= lambda x: torch.abs(-x),
                    ),
            BoundaryCondition(
                    type=BoundaryType.DIRICHLET,
                    location=BoundaryLocation.Z_MAX,
                    value= self.load_data()['bound_out'], #lambda x,y,z : self.u_exact(x , y , z)[-1, :, :],
                    weight=weight,
                    trainable=self.param['TF'],
                    weight_function= lambda x: torch.abs(-x),
                    ),
            BoundaryCondition(
                    type=BoundaryType.DIRICHLET,
                    location=BoundaryLocation.Y_MIN,
                    value= self.load_data()['bound_down'], #lambda x,y,z : self.u_exact(x , y , z)[:, 0, :],
                    weight=weight,
                    trainable=self.param['TF'],
                    weight_function= lambda x: torch.abs(-x),
                    ),
            BoundaryCondition(
                    type=BoundaryType.DIRICHLET,
                    location=BoundaryLocation.Y_MAX,
                    value= self.load_data()['bound_up'], #lambda x,y,z : self.u_exact(x , y , z)[:, -1, :],
                    weight=weight,
                    trainable=self.param['TF'],
                    weight_function= lambda x: torch.abs(-x),
                    ),
            BoundaryCondition(
                    type=BoundaryType.DIRICHLET,
                    location=BoundaryLocation.X_MIN,
                    value= self.load_data()['bound_left'], #lambda x,y,z : self.u_exact(x , y , z)[:, :, 0],
                    weight=weight,
                    trainable=self.param['TF'],
                    weight_function= lambda x: torch.abs(-x),
                    ),
            BoundaryCondition(
                    type=BoundaryType.DIRICHLET,
                    location=BoundaryLocation.X_MAX,
                    value= self.load_data()['bound_right'], #lambda x,y,z : self.u_exact(x , y , z)[:, :, -1],
                    weight=weight,
                    trainable=self.param['TF'],
                    weight_function= lambda x: torch.abs(-x),
                    )
    ]

                



    

