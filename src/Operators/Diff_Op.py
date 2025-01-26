from abc import ABC, abstractmethod
import torch
from enum import Enum
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

class pdeOperator:
    def __init__(self, x=None, y=None, z=None, u=None):
        self.x = x
        self.y = y
        self.z = z
        self.u = u

    def derivative(self, u, x, order=1):
        """Compute the nth-order derivative of u with respect to x."""
        for _ in range(order):
            u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused=True)[0]
        return u

    def gradient(self, u, x, y=None, z=None):
        """Compute the gradient of u in 1D, 2D, or 3D."""
        grad = [self.derivative(u, x)]
        if y is not None:
            grad.append(self.derivative(u, y))
        if z is not None:
            grad.append(self.derivative(u, z))
        return torch.stack(grad, dim=-1)

    def laplacian(self, u, x, y=None, z=None):
        """Compute the Laplacian of u in 1D, 2D, or 3D."""
        lap = self.derivative(u, x , order = 2 )  # Second derivative w.r.t. x
        if y is not None:
            lap += self.derivative(u , y , order = 2)  # Second derivative w.r.t. y
        if z is not None:
            lap += self.derivative(u , z , order = 2)  # Second derivative w.r.t. z
        return lap

    def divergence(self, vec, x, y=None, z=None):
        """Compute the divergence of a vector field in 1D, 2D, or 3D."""
        div = self.derivative(vec[..., 0], x)  # Derivative of the x-component
        if y is not None:
            div += self.derivative(vec[..., 1], y)  # Derivative of the y-component
        if z is not None:
            div += self.derivative(vec[..., 2], z)  # Derivative of the z-component
        return div

    def curl(self, vec, x, y=None, z=None):
        """Compute the curl of a vector field in 2D or 3D."""
        if y is None and z is None:
            raise ValueError("Curl is undefined in 1D.")
        if z is None:  # 2D case
            # Curl is a scalar in 2D: ∂v/∂x - ∂u/∂y
            return self.derivative(vec[..., 1], x) - self.derivative(vec[..., 0], y)
        else:  # 3D case
            # Curl is a vector in 3D
            curl_x = self.derivative(vec[..., 2], y) - self.derivative(vec[..., 1], z)
            curl_y = self.derivative(vec[..., 0], z) - self.derivative(vec[..., 2], x)
            curl_z = self.derivative(vec[..., 1], x) - self.derivative(vec[..., 0], y)
            return torch.stack([curl_x, curl_y, curl_z], dim=-1)

    def hessian(self, u, x, y=None, z=None):
        """Compute the Hessian matrix of u in 1D, 2D, or 3D."""
        hess = [[self.derivative(u, x , order=2)]]
        if y is not None:
            hess[0].append(self.derivative(self.derivative(u, x), y))
            hess.append([self.derivative(self.derivative(u, y), x), self.derivative(u, y , order=2)])
        if z is not None:
            for i in range(len(hess)):
                hess[i].append(self.derivative(self.derivative(u, x), z if i == 0 else y))
            hess.append([self.derivative(self.derivative(u, z), x), self.derivative(self.derivative(u, z), y), self.derivative(u, z , order=2)])
        return torch.stack([torch.stack(row) for row in hess], dim=0)

    def jacobian(self, vec, x, y=None, z=None):
        """Compute the Jacobian matrix of a vector field in 1D, 2D, or 3D."""
        jac = [self.derivative(vec[..., 0], x)]
        if y is not None:
            jac.append(self.derivative(vec[..., 0], y))
            jac.append(self.derivative(vec[..., 1], x))
            jac.append(self.derivative(vec[..., 1], y))
        if z is not None:
            jac.append(self.derivative(vec[..., 0], z))
            jac.append(self.derivative(vec[..., 1], z))
            jac.append(self.derivative(vec[..., 2], x))
            jac.append(self.derivative(vec[..., 2], y))
            jac.append(self.derivative(vec[..., 2], z))
        return torch.stack(jac, dim=-1)

@dataclass
class OperatorConfig:
    """Configuration for PDE operators."""
    operator : Callable
    source_function: Callable
    u_exact: Optional[Callable] = None
    weight: Optional[torch.Tensor] = torch.tensor(1 , dtype=torch.float32)
    trainable: bool = False
    weight_function: Optional[Callable] = None
    pde_loss : Optional[Callable] = F.mse_loss
    adaptive_nodes : Optional[int] = 0
    update_rate : Optional[int] = 0



class PdeLoss:

    def __call__(self, f_exact : torch.Tensor  , f_pred : torch.Tensor, loss_fn : Callable = F.mse_loss):
        return loss_fn(f_exact.squeeze() , f_pred.squeeze())



