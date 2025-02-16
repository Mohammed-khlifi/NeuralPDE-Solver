from src.Operators.Diff_Op import pdeOperator
import torch
d = pdeOperator()

f = lambda X,Y : torch.exp(-10*(X**2 + Y**2))*(-20)*(1-20*X**2) + torch.exp(-10*(X**2 + Y**2))*(-20)*(1-20*Y**2)

operator = lambda u,x,y : d.laplacian(u, x, y)
u_exact = lambda x,y :  torch.exp(-10*(x**2 + y**2))


def load_data():
    inputs = {
    "bound_up": lambda x,y : torch.exp(-10*(y[0,:]**2 + 1)).squeeze(),
    "bound_down": lambda x,y : torch.exp(-10*(y[0,:]**2 + 1)).squeeze(),
    "bound_right": lambda x,y : torch.exp(-10*(x[:,0]**2 + 1)).squeeze(),
    "bound_left": lambda x,y : torch.exp(-10*(x[:,0]**2 + 1)).squeeze(),
    "input": [torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32)],
    }
    return inputs