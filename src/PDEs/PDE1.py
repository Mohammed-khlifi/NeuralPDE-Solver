from src.Operators.Diff_Op import pdeOperator
import torch
d = pdeOperator()

operator = lambda u, x: d.derivative(u, x, order=2) 

def f(x):
    return (-1.6 * (torch.pi**2) * torch.sin(torch.pi * x * 4) - 50 * torch.tanh(5 * x) * (1 - torch.tanh(5 * x)**2))

def u_exact(x):
    return 0.1*torch.sin(torch.pi*x*4) + torch.tanh(5*x)


def load_data():
    inputs = {
    "bound_left": torch.tensor(1.0),
    "bound_right": torch.tensor(-1.0),   
    "input": torch.linspace(-1, 1, 10),
    }
    return inputs