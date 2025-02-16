from src.Operators.Diff_Op import pdeOperator
import torch


d = pdeOperator()
device = "cuda" if torch.cuda.is_available() else "cpu"

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