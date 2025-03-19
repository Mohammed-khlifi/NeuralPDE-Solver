from .basemodel import Basemodel
from .models import PINN_Net, CustomPINN
from Training.trainer import Trainer
from Operators.Bound_Op import BoundaryCondition, BoundaryLocation, BoundaryType    
from Operators import pdeOperator , OperatorConfig
import torch
import typing as tp



class PINNModel(Basemodel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

        self.model = PINN_Net(1, 1, self.param['N_hidden'], self.param['N_layers'])
        
        self.pde_configurations = OperatorConfig(
            operator=self.operator,
            weight= torch.tensor(self.param['weight']),
            source_function=self.f,
            u_exact=self.u_exact,
            trainable=self.param['TF'],
            weight_function= lambda x : torch.exp(-x) ,
            pde_loss = None,
            adaptive_nodes = self.param['adaptive_nodes'],
            update_rate = self.param['update_rate'],

        )
        
        boundary_names = ['bound_left', 'bound_right']
        locations = [BoundaryLocation.X_MAX, BoundaryLocation.X_MIN]
                        
        self.boundary_conditions = [
                BoundaryCondition(
                    type=BoundaryType.DIRICHLET,
                    location=location,    
                    value= self.load_data()[boundarie_name] ,
                    weight= torch.tensor(1.0),#torch.tensor(self.param['weight']),
                    trainable=self.param['TF'],
                    weight_function= lambda x: torch.exp(-x),

                ) for boundarie_name, location in zip(boundary_names, locations)
            ]


class PINNModel_2D(Basemodel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

        self.model =  PINN_Net(2, 1, self.param['N_hidden'], self.param['N_layers'])
        self.pde_configurations = OperatorConfig(
            operator=self.operator,
            weight= torch.tensor(self.param['weight']),
            source_function=self.f,
            u_exact=self.u_exact,
            trainable=self.param['TF'],
            weight_function= lambda x : torch.exp(-x) ,
            pde_loss = None,
            adaptive_nodes = self.param['adaptive_nodes'],
            update_rate = self.param['update_rate'],

        )
        weight= torch.tensor(self.param['weight'])
        boundary_names = ['bound_up', 'bound_down', 'bound_left', 'bound_right']
        locations = [BoundaryLocation.Y_MAX, BoundaryLocation.Y_MIN, BoundaryLocation.X_MIN, BoundaryLocation.X_MAX]
        self.boundary_conditions = [
                BoundaryCondition(
                        type=BoundaryType.DIRICHLET,
                        location=location,
                        value= self.load_data()[boundary_name] ,#lambda x,y : torch.exp(-10*(y[0,:]**2 + 1)).squeeze(),
                        weight= weight,
                        trainable=self.param['TF'],
                        weight_function= lambda x : torch.exp(x),
                        )  for boundary_name, location in zip(boundary_names, locations)   
        ]



class PINNModel_3D(Basemodel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

        self.model = PINN_Net(3, 1, self.param['N_hidden'], self.param['N_layers'])
        self.pde_configurations = OperatorConfig(
            operator=self.operator,
            weight= torch.tensor(self.param['weight']),
            source_function=self.f,
            u_exact=self.u_exact,
            trainable=self.param['TF'],
            weight_function= lambda x : torch.exp(-x) ,
            pde_loss = None,
            adaptive_nodes = self.param['adaptive_nodes'],
            update_rate = self.param['update_rate'],

        )
        weight= torch.tensor(self.param['weight'])
        boundary_names = ['bound_up', 'bound_down', 'bound_left', 'bound_right', 'bound_in', 'bound_out']
        locations = [BoundaryLocation.Y_MAX, BoundaryLocation.Y_MIN, BoundaryLocation.X_MIN, BoundaryLocation.X_MAX, BoundaryLocation.Z_MIN, BoundaryLocation.Z_MAX]
        self.boundary_conditions = [
                BoundaryCondition(
                        type=BoundaryType.DIRICHLET,
                        location=location,
                        value= self.load_data()[boundary_name] ,#lambda x,y : torch.exp(-10*(y[0,:]**2 + 1)).squeeze(),
                        weight= weight,
                        trainable=self.param['TF'],
                        weight_function= lambda x : torch.exp(x),
                        )  for boundary_name, location in zip(boundary_names, locations)   
        ]

                



    

