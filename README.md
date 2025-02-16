
# Physics-Informed Neural Networks (PINNs) Framework

## Directory Structure

project/ ├── src/ │ ├── Models/ │ │ ├── BaseModel.py # Base model implementation │ │ ├── D1_PINNModel.py # 1D PINN implementation │
│ └── callmodel.py # Model factory │ ├── PDEs/ │
│ └── callPDE.py # PDE definitions │ ├── Operators/ │ │ ├── Diff_Op.py # Differential operators │
│ └── Bound_Op.py # Boundary conditions │ ├── Losses/ │ 
│ └── PINN_losses.py # Loss functions │ ├── Training/ │
│ └── trainer.py # Training logic │ └── utils/ 
└── normalizer.py # Data normalization ├── tests/ # Test suite ├── configs/ # Configuration files 
└── main.py # Entry point

##Quick start
python main.py --model_name 1D_PINNModel --PDE PDE1 --epochs 1000

