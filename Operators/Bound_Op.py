import torch
import torch.nn.functional as F
from enum import Enum
from typing import Dict, Union, Callable, Tuple, List
from dataclasses import dataclass

class BoundaryType(Enum):
    """Enumeration of supported boundary condition types."""
    DIRICHLET = "Dirichlet"
    NEUMANN = "Neumann"

class BoundaryLocation(Enum):
    """Enumeration of supported boundary locations."""
    X_MIN = "x_min"
    X_MAX = "x_max"
    Y_MIN = "y_min"
    Y_MAX = "y_max"
    Z_MIN = "z_min"
    Z_MAX = "z_max"


@dataclass
class BoundaryCondition:
    """Data class for boundary condition specification."""
    type: BoundaryType
    location: BoundaryLocation
    value: Union[float, torch.Tensor, Callable]
    weight: float = 1.0,
    trainable: bool = False,
    weight_function: Callable = lambda x: torch.exp(-x)

    def __post_init__(self):
        if not isinstance(self.type, BoundaryType):
            raise ValueError(f"Invalid boundary type: {self.type}")
        if not isinstance(self.location, BoundaryLocation):
            raise ValueError(f"Invalid boundary location: {self.location}")



class BoundaryExtractor:
    """Handles extraction of boundary values from tensors."""
    
    @staticmethod
    def get_boundary_slice(ndim: int, location: BoundaryLocation) -> Tuple[Union[slice, int], ...]:
        """Generate slice tuple for boundary extraction."""
        slices = {
            1: {
                BoundaryLocation.X_MIN: (0,),
                BoundaryLocation.X_MAX: (-1,),
            },
            2: {
                BoundaryLocation.X_MIN: (0, slice(None)),
                BoundaryLocation.X_MAX: (-1, slice(None)),
                BoundaryLocation.Y_MIN: (slice(None), 0),
                BoundaryLocation.Y_MAX: (slice(None), -1),
            },
            3: {
                BoundaryLocation.X_MIN: (0, slice(None), slice(None)),
                BoundaryLocation.X_MAX: (-1, slice(None), slice(None)),
                BoundaryLocation.Y_MIN: (slice(None), 0, slice(None)),
                BoundaryLocation.Y_MAX: (slice(None), -1, slice(None)),
                BoundaryLocation.Z_MIN: (slice(None), slice(None), 0),
                BoundaryLocation.Z_MAX: (slice(None), slice(None), -1),
            }
        }
        
        if ndim not in slices or location not in slices[ndim]:
            raise ValueError(f"Unsupported: {ndim}D and {location}")
        return slices[ndim][location]

    @classmethod
    def extract_boundary(cls, u: torch.Tensor, location: BoundaryLocation) -> torch.Tensor:
        """Extract boundary values using efficient slicing."""
        ndim = u.ndim
        boundary_slice = cls.get_boundary_slice(ndim, location)
        return u[boundary_slice]


class BoundaryLoss(BoundaryExtractor):
    def compute_neumann_loss(self, u_pred: torch.Tensor, bc: BoundaryCondition , coords = None ) -> torch.Tensor:
        grad = torch.autograd.grad(u_pred.sum(), u_pred, create_graph=True)[0]
        boundary_grad = self.extract_boundary(grad, bc.location)
        return bc.weight * F.mse_loss(boundary_grad, bc.value)

    def process_boundary_value(self ,bc, coords):
        try:
            if callable(bc.value):
                boundary = bc.value(*coords)
            elif isinstance(bc.value, (int, float)):
                boundary = torch.tensor(bc.value, dtype=torch.float32)
            elif isinstance(bc.value, torch.Tensor):
                boundary = bc.value
            else:
                raise ValueError(f"Unsupported boundary value type: {type(bc.value)}")
            
            return boundary
        except Exception as e:
            raise ValueError(f"Error processing boundary value: {str(e)}")
    
    def __call__(self, u_pred: torch.Tensor, bc: BoundaryCondition , coords = None) -> torch.Tensor:
        boundarie = self.process_boundary_value(bc, coords)

        if not boundarie.shape == self.extract_boundary(u_pred, bc.location).shape:
            raise ValueError("Boundary value shape mismatch")
            
        if bc.type == BoundaryType.DIRICHLET:
            if bc.weight_function is not None:
                if not isinstance(bc.weight , torch.Tensor):
                    bc.weight = torch.tensor(bc.weight, dtype=torch.float32)     
                return bc.weight_function(bc.weight) * F.mse_loss(
                    self.extract_boundary(u_pred, bc.location),
                    boundarie
                )
            return bc.weight * F.mse_loss(
                self.extract_boundary(u_pred, bc.location),
                boundarie
            ) + torch.abs(bc.weight)
        elif bc.type == BoundaryType.NEUMANN:
            return self.compute_neumann_loss(u_pred, bc)
        else:
            raise NotImplementedError(f"{bc.type} not supported")
        

        
