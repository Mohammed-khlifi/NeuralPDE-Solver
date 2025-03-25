
## PINNs Results

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
      <td><strong>Domain L2 Error</strong></td>
      <td>3.24×10<sup>0</sup></td>
      <td>3.65×10<sup>0</sup></td>
      <td>8.44×10<sup>-2</sup></td>
      <td>5.13×10<sup>-1</sup></td>
      <td>4.42×10<sup>-1</sup></td>
      <td>1.15×10<sup>-1</sup></td>
      <td>6.83×10<sup>-4</sup></td>
      <td>2.73×10<sup>-3</sup></td>
    </tr>
    <tr>
      <td><strong>Domain MSE Error</strong></td>
      <td>1.05×10<sup>-3</sup></td>
      <td>1.33×10<sup>-3</sup></td>
      <td>7.13×10<sup>-7</sup></td>
      <td>2.63×10<sup>-5</sup></td>
      <td>1.55×10<sup>-1</sup></td>
      <td>1.05×10<sup>-2</sup></td>
      <td>3.71×10<sup>-7</sup></td>
      <td>5.93×10<sup>-6</sup></td>
    </tr>
    <tr>
      <td colspan="9"><strong>Bottom Boundary</strong></td>
    </tr>
    <tr>
      <td><strong>Bottom L2</strong></td>
      <td>3.13×10<sup>-2</sup></td>
      <td>1.13×10<sup>-2</sup></td>
      <td>1.34×10<sup>-2</sup></td>
      <td>1.14×10<sup>-2</sup></td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
    </tr>
    <tr>
      <td><strong>Bottom MSE</strong></td>
      <td>9.82×10<sup>-6</sup></td>
      <td>1.27×10<sup>-6</sup></td>
      <td>1.79×10<sup>-6</sup></td>
      <td>1.30×10<sup>-6</sup></td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
    </tr>
    <tr>
      <td colspan="9"><strong>Top Boundary</strong></td>
    </tr>
    <tr>
      <td><strong>Top L2</strong></td>
      <td>4.03×10<sup>-2</sup></td>
      <td>2.80×10<sup>-2</sup></td>
      <td>1.21×10<sup>-2</sup></td>
      <td>2.34×10<sup>-2</sup></td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
    </tr>
    <tr>
      <td><strong>Top MSE</strong></td>
      <td>1.62×10<sup>-5</sup></td>
      <td>7.83×10<sup>-6</sup></td>
      <td>1.46×10<sup>-6</sup></td>
      <td>5.49×10<sup>-6</sup></td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
    </tr>
    <tr>
      <td colspan="9"><strong>Left Boundary</strong></td>
    </tr>
    <tr>
      <td><strong>Left L2</strong></td>
      <td>1.16×10<sup>-2</sup></td>
      <td>9.15×10<sup>-3</sup></td>
      <td>6.46×10<sup>-3</sup></td>
      <td>1.83×10<sup>-2</sup></td>
      <td>5.26×10<sup>-4</sup></td>
      <td>4.29×10<sup>-4</sup></td>
      <td>1.06×10<sup>-3</sup></td>
      <td>1.57×10<sup>-4</sup></td>
    </tr>
    <tr>
      <td><strong>Left MSE</strong></td>
      <td>1.36×10<sup>-6</sup></td>
      <td>8.37×10<sup>-7</sup></td>
      <td>4.17×10<sup>-7</sup></td>
      <td>3.37×10<sup>-6</sup></td>
      <td>2.76×10<sup>-7</sup></td>
      <td>1.84×10<sup>-7</sup></td>
      <td>1.11×10<sup>-6</sup></td>
      <td>2.46×10<sup>-8</sup></td>
    </tr>
    <tr>
      <td colspan="9"><strong>Right Boundary</strong></td>
    </tr>
    <tr>
      <td><strong>Right L2</strong></td>
      <td>9.94×10<sup>-3</sup></td>
      <td>1.20×10<sup>-2</sup></td>
      <td>2.68×10<sup>-2</sup></td>
      <td>1.96×10<sup>-2</sup></td>
      <td>7.01×10<sup>-4</sup></td>
      <td>4.26×10<sup>-3</sup></td>
      <td>1.05×10<sup>-3</sup></td>
      <td>1.88×10<sup>-4</sup></td>
    </tr>
    <tr>
      <td><strong>Right MSE</strong></td>
      <td>9.89×10<sup>-7</sup></td>
      <td>1.45×10<sup>-6</sup></td>
      <td>7.20×10<sup>-6</sup></td>
      <td>3.83×10<sup>-6</sup></td>
      <td>4.86×10<sup>-7</sup></td>
      <td>1.79×10<sup>-5</sup></td>
      <td>1.08×10<sup>-6</sup></td>
      <td>3.50×10<sup>-8</sup></td>
    </tr>
  </tbody>
</table>


## Neural Operators results

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

### 1.0 Cloning the repository
This repository uses Git submodules, so it's essential to clone it recursively to ensure that all required submodules are properly initialized and updated.

To clone the repository with submodules, run:

```bash
git clone --recursive 
```
If you've already cloned the repository without the ```--recursive``` flag, initialize and update the submodules by executing
```bash
git submodule update --init --recursive
```

### 1.1 Solving a Single PDE

```bash
python main.py --model_type PINN --model_name <MODEL_NAME> --PDE <PDE_NAME> --config <CONFIG_FILE>
```

Options :
- model_type PINN: Specifies that the chosen model is a physics-informed neural network variant.
- model_name <MODEL_NAME>: User-defined name for the model (e.g., 1D_PINNmodel).
- PDE <PDE_NAME>: The PDE to be solved (e.g., Poisson, Burgers, etc.).
- config <CONFIG_FILE>: Path to a configuration file specifying hyperparameters  


### Training a single model on a single dataset

```bash
python main.py --model_type NO --model_name <MODEL_NAME> --Dataset <DATASET_NAME> --config <CONFIG_FILE>
```
Options :
- model_type NO: Indicates a Neural Operator–based model.
- model_name <MODEL_NAME>: User-defined identifier for the model (e.g., FNO).
- Dataset <DATASET_NAME>: Dataset name to be used for training (e.g., darcy_flow , Poisson).
- config <CONFIG_FILE>: Path to the config file. This file can include settings for batch size, number of training epochs, or resolution details.

### Adding a new PDE

in PDE.py add the following 

- the PDE name
- the PDE operator
- the boundary consitions
- the coordinates
- exact solution (None if not available)

### adding New dataset

in Dataloader.py add 
- Dataset name 
- return Dataloader and the Testloaders (the code is designed to test the models on different resolutions)


### adding New Model

all models should inherit from besemodel and NO_basemodel for databased models

define the following :

- def __init__ : defining the model
- def fit(dataloader , testloders): 


Add your ```<model>.py``` file to the models directory and do not forget to update the Models/__init__.py file.
