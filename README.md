
# **Physics-Informed Neural Networks (PINNs) Framework**

## **📖 Overview**
This repository provides a modular and extensible framework for solving Partial Differential Equations (**PDEs**) using **Physics-Informed Neural Networks (PINNs)**. The framework supports **1D, 2D and 3D**, and **scalable extensions**, incorporating **differential operators, boundary conditions, adaptive collocation strategies, and custom loss functions**. 

Additionally, this framework allows benchmarking PINNs against **Neural Operators** such as **Fourier Neural Operator (FNO)** and **Physics-Informed Neural Operator (PINO)**, enabling comparisons across **different learning paradigms**.

---

## **🔹 Features**
✔ **Modular & Scalable** – Easily extendable for different PDEs and PINN architectures.  
✔ **Customizable PINN Models** – Supports **1D, 2D and 3D PINNs** with adaptive techniques.  
✔ **Flexible PDE Definitions** – Users can define PDE operators and boundary conditions dynamically.  
✔ **Adaptive Collocation Sampling** – Focuses training on regions with high PDE residuals.  
✔ **Custom Loss Functions** – Includes physics-based loss terms and trainable weighting strategies.  
✔ **Configurable via YAML** – Hyperparameters and model settings are easily adjustable.  
✔ **Benchmarking & Experiment Tracking** – Supports **comparison between PINNs, FNOs, and PINOs**.  

---

## **📂 Project Structure**

project/
│── src/
│   ├── Models/
│   │   ├── BaseModel.py         # Generalized base model for PINNs
│   │   ├── PINNModelq.py      # 1D PINN implementation
│   ├── PDEs/
│   │   ├── PDE1.py              # 1D PDE definitions
│   │   ├── PDE2D_example.py     # 2D PDE definitions
│   ├── Operators/
│   │   ├── Diff_Op.py           # Differential operators (gradients, Laplacians)
│   │   ├── Bound_Op.py          # Boundary condition definitions
│   ├── Losses/
│   │   ├── PINN_losses.py       # Loss functions for physics-informed training
│   ├── Training/
│   │   ├── trainer.py           # Training logic for PINNs

│── tests/                        # Unit and integration tests
│── configs/                      # YAML-based configuration files
│── main.py                        # Entry point for training and evaluation



## **🚀 Quick Start**
### **🔹 Run a 1D PINN Model**
Train a **1D PINN** on **PDE1** with **1000 epochs**:
```bash
python main.py --model_name 1D_PINNModel --PDE PDE1 --epochs 1000


