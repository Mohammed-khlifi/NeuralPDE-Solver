# src/__init__.py
import sys
import os

# Add the path first, before any imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Then do your imports
from src.Training.trainer import Trainer
from src.Training.solver import Solver
from src.Training.NeuralPINN import PINO
from src.Models.models import PINN_Net, CustomPINN

# Define what should be available when importing from src
__all__ = ['Trainer', 'Solver', 'PINO', 'PINN_Net', 'CustomPINN']