from src.Operators.Diff_Op import pdeOperator
import torch

d = pdeOperator()

def load_PDE(PDE_name):
    if PDE_name == 'PDE1':
        operator = lambda u, x: d.derivative(u, x, order=2) 
        f = lambda x: (-1.6 * (torch.pi**2) * torch.sin(torch.pi * x * 4) - 50 * torch.tanh(5 * x) * (1 - torch.tanh(5 * x)**2))
        u_exact = lambda x: 0.1*torch.sin(torch.pi*x*4) + torch.tanh(5*x)
        def load_data():
            inputs = {
            "bound_left": torch.tensor(1.0),
            "bound_right": torch.tensor(-1.0),   
            "input": torch.linspace(-1, 1, 10),
            }
            return inputs
        return operator, f, u_exact, load_data
    elif PDE_name == 'PDE2':
        f = lambda x,y : -2 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        operator = lambda u,x,y : d.laplacian(u , x , y)
        u_exact = lambda x,y : torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        def load_data():
            inputs = {
            "bound_up": lambda x,y : torch.exp(-10*(y[0,:]**2 + 1)).squeeze(),
            "bound_down": lambda x,y : torch.exp(-10*(y[0,:]**2 + 1)).squeeze(),
            "bound_right": lambda x,y : torch.exp(-10*(x[:,0]**2 + 1)).squeeze(),
            "bound_left": lambda x,y : torch.exp(-10*(x[:,0]**2 + 1)).squeeze(),
            "input": [torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10)],
            }
            return inputs
        return operator, f, u_exact, load_data
    elif PDE_name == 'PDE3':
        f = lambda x,y,z : -3 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z)
        operator= lambda u,x,y,z : d.laplacian(u , x , y , z)
        u_exact = lambda x,y,z : torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z)
        def load_data():
            inputs = {
            "bound_in": lambda x,y,z : u_exact(x , y , z)[0, :, :],
            "bound_out": lambda x,y,z : u_exact(x , y , z)[-1, :, :],
            "bound_down": lambda x,y,z : u_exact(x , y , z)[:, 0, :],
            "bound_up": lambda x,y,z : u_exact(x , y , z)[:, -1, :],
            "bound_left": lambda x,y,z : u_exact(x , y , z)[:, :, 0],
            "bound_right": lambda x,y,z : u_exact(x , y , z)[:, :, -1],
            "input": [torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10)],
            }
            return inputs
        return operator, f, u_exact, load_data
    