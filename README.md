# PINNs and Neural Operators for PDE Solving

This repository provides implementations of **Physics-Informed Neural Networks (PINNs)** and **Neural Operators (NOs)** for solving Partial Differential Equations (PDEs). It includes multiple models and training configurations to handle different PDE types efficiently.

## PINNs Results

The table below presents the performance of different **PINN variants** on **1D and 2D Poisson equations** using **L2 error** and **Mean Squared Error (MSE)** as evaluation metrics.

<table>
  <thead>
    <tr>
      <th rowspan="2">Metric</th>
      <th colspan="4">2D Poisson</th>
      <th colspan="4">1D Poisson</th>
    </tr>
    <tr>
      <th>PINN</th>
      <th>AW-PINN</th>
      <th>AC-PINN</th>
      <th>ACAW-PINN</th>
      <th>PINN</th>
      <th>AW-PINN</th>
      <th>AC-PINN</th>
      <th>ACAW-PINN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Domain L2 Error</td>
      <td>4.47×10<sup>0</sup></td>
      <td>9.03×10<sup>−1</sup></td>
      <td>1.32×10<sup>−2</sup></td>
      <td>1.22×10<sup>−1</sup></td>
      <td>4.42×10<sup>−1</sup></td>
      <td>1.15×10<sup>−1</sup></td>
      <td>6.83×10<sup>−4</sup></td>
      <td>2.73×10<sup>−3</sup></td>
    </tr>
    <tr>
      <td>Domain MSE Error</td>
      <td>2.24×10<sup>−3</sup></td>
      <td>8.16×10<sup>−5</sup></td>
      <td>1.75×10<sup>−6</sup></td>
      <td>1.48×10<sup>−6</sup></td>
      <td>1.55×10<sup>−1</sup></td>
      <td>1.05×10<sup>−2</sup></td>
      <td>3.71×10<sup>−7</sup></td>
      <td>5.93×10<sup>−6</sup></td>
    </tr>
    <tr>
      <th colspan="9" style="text-align:left;">Bottom Boundary</th>
    </tr>
    <tr>
      <td>Bottom L2</td>
      <td>1.17×10<sup>−2</sup></td>
      <td>6.11×10<sup>−3</sup></td>
      <td>1.96×10<sup>−2</sup></td>
      <td>1.24×10<sup>−2</sup></td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
    </tr>
    <tr>
      <td>Bottom MSE</td>
      <td>1.38×10<sup>−6</sup></td>
      <td>3.74×10<sup>−7</sup></td>
      <td>3.83×10<sup>−6</sup></td>
      <td>1.53×10<sup>−6</sup></td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
    </tr>
    <tr>
      <th colspan="9" style="text-align:left;">Top Boundary</th>
    </tr>
    <tr>
      <td>Top L2</td>
      <td>1.35×10<sup>−2</sup></td>
      <td>7.27×10<sup>−3</sup></td>
      <td>1.45×10<sup>−2</sup></td>
      <td>3.98×10<sup>−3</sup></td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
    </tr>
    <tr>
      <td>Top MSE</td>
      <td>1.83×10<sup>−6</sup></td>
      <td>5.28×10<sup>−7</sup></td>
      <td>2.11×10<sup>−6</sup></td>
      <td>1.58×10<sup>−7</sup></td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
    </tr>
    <tr>
      <th colspan="9" style="text-align:left;">Left Boundary</th>
    </tr>
    <tr>
      <td>Left L2</td>
      <td>1.37×10<sup>−2</sup></td>
      <td>8.11×10<sup>−3</sup></td>
      <td>2.29×10<sup>−2</sup></td>
      <td>5.87×10<sup>−3</sup></td>
      <td>5.26×10<sup>−4</sup></td>
      <td>4.29×10<sup>−4</sup></td>
      <td>1.06×10<sup>−3</sup></td>
      <td>1.57×10<sup>−4</sup></td>
    </tr>
    <tr>
      <td>Left MSE</td>
      <td>1.88×10<sup>−6</sup></td>
      <td>6.58×10<sup>−7</sup></td>
      <td>5.24×10<sup>−6</sup></td>
      <td>3.44×10<sup>−7</sup></td>
      <td>2.76×10<sup>−7</sup></td>
      <td>1.84×10<sup>−7</sup></td>
      <td>1.11×10<sup>−6</sup></td>
      <td>2.46×10<sup>−8</sup></td>
    </tr>
    <tr>
      <th colspan="9" style="text-align:left;">Right Boundary</th>
    </tr>
    <tr>
      <td>Right L2</td>
      <td>8.33×10<sup>−3</sup></td>
      <td>5.50×10<sup>−3</sup></td>
      <td>8.57×10<sup>−3</sup></td>
      <td>1.14×10<sup>−2</sup></td>
      <td>7.01×10<sup>−4</sup></td>
      <td>4.26×10<sup>−3</sup></td>
      <td>1.05×10<sup>−3</sup></td>
      <td>1.88×10<sup>−4</sup></td>
    </tr>
    <tr>
      <td>Right MSE</td>
      <td>6.94×10<sup>−7</sup></td>
      <td>3.02×10<sup>−7</sup></td>
      <td>7.35×10<sup>−7</sup></td>
      <td>1.29×10<sup>−6</sup></td>
      <td>4.86×10<sup>−7</sup></td>
      <td>1.79×10<sup>−5</sup></td>
      <td>1.08×10<sup>−6</sup></td>
      <td>3.50×10<sup>−8</sup></td>
    </tr>
  </tbody>
</table>

**Boundary Errors:**  
Detailed L2 and MSE errors are provided for the **Top, Bottom, Left, and Right boundaries** in the full table above.


## Neural Operators results

The table below presents training and test losses for different **Neural Operator models** on **Darcy Flow** and **Poisson equations**.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">Darcy</th>
      <th colspan="2">Poisson</th>
    </tr>
    <tr>
      <th>Train Loss</th>
      <th>Test Loss</th>
      <th>Train Loss</th>
      <th>Test Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FNO_model</td>
      <td>$1.73\times10^{-4} \pm 7.57\times10^{-5}$</td>
      <td>$2.24\times10^{-2}$</td>
      <td>$3.28\times10^{-5} \pm 1.96\times10^{-5}$</td>
      <td>$3.84\times10^{-4} \pm 2.54\times10^{-4}$</td>
    </tr>
    <tr>
      <td>CNN_model</td>
      <td>$9.35\times10^{-3} \pm 3.31\times10^{-4}$</td>
      <td>$4.30\times10^{-2}$</td>
      <td>$2.05\times10^{-5} \pm 1.60\times10^{-5}$</td>
      <td>$1.35\times10^{-4} \pm 1.52\times10^{-4}$</td>
    </tr>
    <tr>
      <td>UNO_model</td>
      <td>$7.77\times10^{-4} \pm 2.83\times10^{-4}$</td>
      <td>$3.02\times10^{-2}$</td>
      <td>$7.70\times10^{-5} \pm 4.39\times10^{-5}$</td>
      <td>$3.24\times10^{-4} \pm 3.30\times10^{-4}$</td>
    </tr>
    <tr>
      <td>TFNO_model</td>
      <td>$4.10\times10^{-4} \pm 7.77\times10^{-5}$</td>
      <td>$1.38\times10^{-2}$</td>
      <td>$1.80\times10^{-5} \pm 4.69\times10^{-6}$</td>
      <td>$1.56\times10^{-4} \pm 7.19\times10^{-5}$</td>
    </tr>
    <tr>
      <td>PINO</td>
      <td>$2.09\times10^{-4} \pm 7.41\times10^{-5}$</td>
      <td>$2.10\times10^{-2}$</td>
      <td>$2.51\times10^{-5} \pm 1.59\times10^{-5}$</td>
      <td>$6.66\times10^{-5} \pm 3.52\times10^{-5}$</td>
    </tr>
  </tbody>
</table>

---


## 1. How to Use

### 1.1 Installation
#### Step 1: Cloning the repository
This repository uses Git submodules, so it's essential to clone it recursively to ensure that all required submodules are properly initialized and updated.

To clone the repository with submodules, run:

```bash
git clone --recursive https://github.com/Mohammed-khlifi/NeuralPDE-Solver.git
```
If you've already cloned the repository without the ```--recursive``` flag, initialize and update the submodules by executing
```bash
git submodule update --init --recursive 
```

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Install the library

- Developement mode:
```bash
python3 -m pip install --user -e .
```

- Install globaly
```bash
pip install .
```

### 1.2 Solving a Single PDE

To solve a PDE using Physics-Informed Neural Networks (PINNs), run:

```bash
python main.py --model_type PINN --model_name <MODEL_NAME> --PDE <PDE_NAME> --config <CONFIG_FILE>
```

Options :
- `--model_type PINN`: Specifies that the chosen model is a physics-informed neural network variant.
- `--model_name <MODEL_NAME>`: User-defined name for the model (e.g., `1D_PINNmodel`).
  - `1D_PINNmodel`: for 1D equations
  - `2D_PINNmodel`: for 2D equations
  - `3D_PINNmodel`: for 3D equations
- `--PDE <PDE_NAME>`: The PDE to be solved (e.g., `Poisson`, etc.).
  - `1DPoisson`
  - `2DPoisson`
  - `3DPoisson`
- `--config <CONFIG_FILE>`: Path to the yaml configuration file specifying hyperparameters
- (Optional) `--adaptive_weights` (1 , 0)
- (Optional) `--AC` 
- (Optional) `--update_rate` 
- ... and so on, use `python main.py -h` to see all the other possible parameters

### 1.3 Training a single model on a single dataset

Install first `neuraloperator` package
```bash
cd neuraloperator
pip3 install -e .
```

To train a Neural Operator-based model, run:

```bash
python main.py --model_type NO --model_name <MODEL_NAME> --Dataset <DATASET_NAME> --config <CONFIG_FILE>
```
Options :
- `--model_type NO`: Indicates a Neural Operator–based model.
- `--model_name <MODEL_NAME>`: User-defined identifier for the model (e.g., `FNO`).
  - `CNN`: Convolution Neural Network
  - `FNO`: Fourier Neural Operator
  - `UNO`: U-shaped Neural Operator
  - `TFNO`: Tensorized Fourier Neural Operator
  - `PINO`: Physics Informed Neural Operator
- `--Dataset <DATASET_NAME>`: Dataset name to be used for training (e.g., `darcy_flow`, `Poisson`).
  - darcy_flow
  - Poisson
- `--config <CONFIG_FILE>`: Path to the yaml configuration file specifying hyperparameters

## 2. Extending the Repository

### 2.1 Adding a new PDE

To introduce a new Partial Differential Equation (PDE), update PDE.py:

1. Define the PDE name.
2. Specify the PDE operator (e.g., `Laplacian` or any custumized operator).
   How to define PDE operator :
   ```bash
   from Operators.Diff_Op import pdeOperator
   D = pdeOperator()
   1D_operator = lambda u, x: D.derivative(u, x, order=2)*u
   2D_operator = lambda u, x, y: D.derivative(u, x, order=2)*u + D.derivative(u, y, order=1)
   3D_operator = lambda u, x, y, z: D.derivative(u, x, order=2) + D.derivative(u, y, order=2) + D.derivative(u, y, order=2) # equivalant to d.laplacian(u, x, y, z) 
   ```
   
4. Add the boundary conditions.
5. Include the coordinate system.
6. Provide the exact solution (if available, otherwise set to `None`).

Example :  
```bash
def load_PDE(PDE_name):
    ## existing PDEs
    if PDE_name == 'New PDE':
        operator = lambda u, x: d.derivative(u, x, order=2)*u # operator example
        f = lambda x: torche.ones_like(x) # source function example
        u_exact = None # if not available return None
        def load_data():
            inputs = {
            "bound_left": torch.tensor(1.0),
            "bound_right": torch.tensor(-1.0),   
            "input": torch.linspace(-1, 1, 10), # coordinates
            }
            return inputs
        return operator, f, u_exact, load_data
```

    
### 2.2 Adding New dataset

**Adding a New Dataset**

To introduce a new dataset, open the `Dataloader.py` file and follow these steps:

1. **Define the Dataset Name**  
   Add a condition for your dataset name in the `load_dataset()` function (or whichever main loader function you are using).

2. **Return DataLoader and TestLoaders**  
   Implement the logic for constructing and returning the training dataloader along with any test dataloaders (possibly at different resolutions).

For example:
```bash
def load_dataset(data_name):
    # ... existing code for other datasets ...

    if data_name == 'New data':
        # Your data loading logic here
        # e.g., read files, preprocess, create dataset objects
        
        train_loader = ...
        test_loaders = {
            'low_res': ...,
            'high_res': ...
            # or any structure you prefer
        }

        return train_loader, test_loaders

```
    
### 2.3 Adding New Model

All new model classes must inherit from:
1. `Basemodel` (for PINNs), or  
2. `NO_Basemodel` (for neural-operator-based models).  

In other words, **all PINN-type models** derive from `Basemodel`, and **Neural Operator models** (those trained primarily on data) derive from `NO_Basemodel`.

### Steps to Add a Model

1. **Create the Model Class**  
   Define your model in a new file named `<model_name>.py` inside the `models/` directory.  
   ```python
   from .basemodel import Basemodel  # or NO_Basemodel if it's a neural operator

   class NewModel(Basemodel):
       def __init__(self, ...):
           super().__init__()
           # Define layers and architecture here
    ```

2. Implement the fit Method

```python
def fit(self, dataloader, testloaders):
    # Implement your custom training loop
    pass

```

3. Add the`<model>.py` file to the `Models/` directory.
4. Update `models/__init__.py` to register your model.
 ````bash
  elif model_name == 'CNN':
      from .CNN_model import CNN_model
      return CNN_model
 ````
