import torch
from torch.utils.data import Dataset, DataLoader, random_split
from Operators.Diff_Op import pdeOperator
import numpy as np
import torch

d = pdeOperator()
derivation = d.derivative
class loadDataset(Dataset):
    """Preloaded Dataset for Physics-Informed Neural Networks"""
    def __init__(self, x, y, p, u_exact=None, operator=None):
        self.samples = []  # Preload all samples here
        for idx in range(len(p)):

            exact = u_exact(x, y, p[idx])
            f = operator(exact, x, y)

            sample = {
                'y': exact.reshape(1, exact.shape[0], exact.shape[1]),
                'x': f.reshape(1, f.shape[0], f.shape[1])
            }
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def creat_coordenates(Nx, Ny , x_min = -1, x_max = 1, y_min = -1, y_max =1 , device = 'cpu'):
    x = torch.linspace(-1, 1, Nx, requires_grad=True , device = device)
    y = torch.linspace(-1, 1, Ny, requires_grad=True , device = device)
    x, y = torch.meshgrid(x, y)
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    return x, y
def u_exact(x, y, p):
    return torch.exp(p * (-x**2 - y**2))

def operator(u, x, y):
    return derivation(u, x, order=2) + derivation(u, y, order=2)

def create_dataset(Nx, Ny, p_values, u_exact, operator):
    X, Y = creat_coordenates(Nx, Ny)
    dataset = loadDataset(X, Y, p_values, u_exact, operator)
    return dataset

def split_dataset(dataset, train_size = 0.8):
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def Train_Test_loaders(resolution=16, p_min=5, p_max=25 ,n_samples=500 ,u_exact = u_exact , operator = operator ,train_size = 0.8, batch_size = 32):
    Nx, Ny = resolution , resolution
    p_values = np.linspace(p_min, p_max, n_samples)
    dataset = create_dataset(Nx, Ny, p_values, u_exact, operator)
    train_dataset, test_dataset = split_dataset(dataset, train_size = 0.8)
    train_loader = DataLoader(train_dataset , batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset , batch_size=batch_size, shuffle=False)

    test_loaders = {resolution: test_loader}
    train_loaders = {resolution: train_loader}
    return train_loaders, test_loaders




def load_dataset(data_name):
    if data_name == 'darcy_flow':
        from neuraloperator.neuralop.data.datasets import load_darcy_flow_small
        train_loader, test_loaders, _ = load_darcy_flow_small(
            n_train=500, batch_size=32,
            test_resolutions=[16, 32], n_tests=[100, 50],
            test_batch_sizes=[32, 32],
        )
        return train_loader, test_loaders
    
    elif data_name == 'Poisson':
        train_loaders, test_loaders = Train_Test_loaders(resolution=16, p_min=3, p_max=60 ,n_samples=500 ,u_exact = u_exact , operator = operator ,train_size = 0.8, batch_size = 32)
        train_loader = train_loaders[16]
        return train_loader, test_loaders
    else:
        raise ValueError(f'Unknown dataset: {data_name}')