# src/__init__.py

#imports
from .Training import Trainer
from .Training import Solver
from .Training import PINO
from .Operators import BoundaryCondition , BoundaryType , BoundaryLocation , BoundaryLoss
from .Operators import pdeOperator , OperatorConfig
from .Models import PINN_Net, CustomPINN
from .Losses import LpLoss, VariationalLoss

import sys
sys.path.append(r"C:\Users\mohammed\OneDrive\Documents\QFM -S2\Solving PDE's using ANN\Solving-PDE-s-using-neural-network\src\neuraloperator")

from .neuraloperator.neuralop.models import FNO
from .neuraloperator.neuralop import Trainer
from .neuraloperator.neuralop.training import AdamW
from .neuraloperator.neuralop.data.datasets import load_darcy_flow_small
from .neuraloperator.neuralop.utils import count_model_params
from .neuraloperator.neuralop import LpLoss, H1Loss

# Define what should be available when importing from src
__all__ = ['Trainer', 'Solver', 'PINO', 'PINN_Net', 'CustomPINN' , 'LpLoss', 'VariationalLoss', 'BoundaryCondition' , 'BoundaryType' , 'BoundaryLocation' , 'BoundaryLoss' , 'pdeOperator' , 'OperatorConfig']