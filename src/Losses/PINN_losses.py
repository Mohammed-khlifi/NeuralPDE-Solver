from numpy.polynomial.legendre import legval
from scipy.special import comb
from scipy.special import gamma
import torch




class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    

class VariatinalLoss(object):
    def __init__(self, alpha=1.0, beta=1.0):
        super(VariatinalLoss, self).__init__()


    def P_k(self, x, k):
        """Generate the Legendre polynomial P_k(x)."""
        coeffs = [0] * (k + 1)
        coeffs[-1] = 1
        return legval(x, coeffs)

    def v(self, x, k):
        P_k_plus = self.P_k(x, k + 1)
        P_k_minus = self.P_k(x, k - 1)
        v_k = P_k_plus - P_k_minus
        return v_k

    def simpsons_rule(self, f, x):
        """Compute integral via Simpson's rule for 1D arrays."""
        N = x.size(0) - 1
        h = (x[-1] - x[0]) / N
        return h / 3 * (f[0] + f[-1] + 4 * f[1::2].sum() + 2 * f[2:-1:2].sum())
    
    def V_loss(self, f_pred , f, degree=8):
        """Compute variational loss for 1D problems."""

        # Compute integral of ((f_pred - f)*v_k)^2 over domain
        x = torch.linspace(-1, 1, f_pred.size(0))
        variational_term = [
            self.simpsons_rule(((f_pred - f) * self.v(x, k))**2, x.squeeze())
            for k in range(1, degree + 1)
        ]
        return sum(variational_term) / degree
    
    def __call__(self, f_pred, f):
        return self.V_loss(f_pred, f)
    


class VariationalLoss:
    def __init__(self, dim, dx , reduction="mean"):
        """
        Variational loss for PINNs.

        Args:
            dim (int): Dimensionality of the problem (1, 2, or 3).
            reduction (str): How to reduce the loss ('mean' or 'sum').
        """
        self.dim = dim
        self.reduction = reduction
        self.dx = dx

    def compute_residual(self, f, f_pred):
        """
        Compute the residual between the true function f and its prediction f_pred.

        Args:
            f (torch.Tensor): Exact solution or source term.
            f_pred (torch.Tensor): Predicted solution or source term.

        Returns:
            torch.Tensor: Residual (f - f_pred).
        """
        return f - f_pred

    def integrate(self, residual, dx):
        """
        Integrate the residual over the domain.

        Args:
            residual (torch.Tensor): Residual of the PDE.
            dx (float): Spacing in each dimension (scalar for 1D, tuple for 2D/3D).

        Returns:
            torch.Tensor: Integral of the residual squared.
        """
        if self.dim == 1:
            integral = torch.sum(residual**2) * dx
        elif self.dim == 2:
            integral = torch.sum(residual**2) * dx[0] * dx[1]
        elif self.dim == 3:
            integral = torch.sum(residual**2) * dx[0] * dx[1] * dx[2]
        else:
            raise ValueError("Only 1D, 2D, and 3D problems are supported.")

        return integral

    def forward(self, f, f_pred):
        """
        Compute the variational loss.

        Args:
            f (torch.Tensor): Exact solution or source term.
            f_pred (torch.Tensor): Predicted solution or source term.
            dx (float or tuple): Grid spacing in each dimension.

        Returns:
            torch.Tensor: Variational loss.
        """
        dx = self.dx
        # Compute the residual
        residual = self.compute_residual(f, f_pred)

        # Integrate the squared residual
        integral = self.integrate(residual, dx)

        # Reduce the loss
        if self.reduction == "mean":
            return integral / f.numel()
        elif self.reduction == "sum":
            return integral
        else:
            raise ValueError("Reduction must be 'mean' or 'sum'.")

class VariationalLoss:
    def __init__(self, dim, dx, reduction="mean", integration_method="simpson"):
        """
        Enhanced variational loss for PINNs with precise integration.

        Args:
            dim (int): Dimensionality of the problem (1, 2, or 3).
            dx (float or tuple): Grid spacing in each dimension.
            reduction (str): How to reduce the loss ('mean', 'sum', or 'none').
            integration_method (str): Method of integration ('simpson', 'trapezoidal', or 'riemann').
        """
        self.dim = dim
        self.reduction = reduction
        
        # Convert dx to tuple if scalar
        self.dx = dx if isinstance(dx, tuple) else (dx,) * dim
        
        # Validate inputs
        if dim not in [1, 2, 3]:
            raise ValueError("Dimension must be 1, 2, or 3")
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")
        if integration_method not in ["simpson", "trapezoidal", "riemann"]:
            raise ValueError("Integration method must be 'simpson', 'trapezoidal', or 'riemann'")
            
        self.integration_method = integration_method

    def compute_residual(self, f, f_pred):
        """
        Compute the residual between the true function f and its prediction f_pred
        with improved numerical stability.

        Args:
            f (torch.Tensor): Exact solution or source term.
            f_pred (torch.Tensor): Predicted solution or source term.

        Returns:
            torch.Tensor: Residual (f - f_pred).
        """
        # Ensure same device and dtype
        if f.device != f_pred.device:
            f_pred = f_pred.to(f.device)
        if f.dtype != f_pred.dtype:
            f_pred = f_pred.to(dtype=f.dtype)
            
        # Compute residual with better numerical stability
        residual = torch.where(
            torch.abs(f) > torch.abs(f_pred),
            f * (1 - f_pred/f),
            f_pred * (f/f_pred - 1)
        )
        
        return residual

    def simpson_integrate_1d(self, values, dx):
        """
        Perform 1D Simpson's rule integration.
        """
        n = len(values)
        if n % 2 == 0:
            n -= 1  # Ensure odd number of points
            
        weights = torch.ones_like(values[:n])
        weights[1:n-1:2] = 4
        weights[2:n-1:2] = 2
        
        return (dx / 3) * torch.sum(weights * values[:n])

    def trapezoidal_integrate_1d(self, values, dx):
        """
        Perform 1D trapezoidal rule integration.
        """
        weights = torch.ones_like(values)
        weights[0] = weights[-1] = 0.5
        return dx * torch.sum(weights * values)

    def integrate(self, residual, dx):
        """
        Integrate the residual over the domain using the specified method.

        Args:
            residual (torch.Tensor): Residual of the PDE.
            dx (tuple): Spacing in each dimension.

        Returns:
            torch.Tensor: Integral of the residual squared.
        """
        squared_residual = residual**2
        
        if self.dim == 1:
            if self.integration_method == "simpson":
                integral = self.simpson_integrate_1d(squared_residual, dx[0])
            elif self.integration_method == "trapezoidal":
                integral = self.trapezoidal_integrate_1d(squared_residual, dx[0])
            else:  # riemann
                integral = torch.sum(squared_residual) * dx[0]
                
        elif self.dim == 2:
            # Apply method along each dimension
            if self.integration_method == "simpson":
                integral = torch.tensor(0., device=residual.device)
                for i in range(residual.shape[0]):
                    row_integral = self.simpson_integrate_1d(squared_residual[i], dx[1])
                    integral += row_integral
                integral = self.simpson_integrate_1d(integral.unsqueeze(0), dx[0])
            else:  # trapezoidal or riemann
                integral = torch.sum(squared_residual) * dx[0] * dx[1]
                
        elif self.dim == 3:
            if self.integration_method == "simpson":
                integral = torch.tensor(0., device=residual.device)
                for i in range(residual.shape[0]):
                    plane_integral = torch.tensor(0., device=residual.device)
                    for j in range(residual.shape[1]):
                        row_integral = self.simpson_integrate_1d(squared_residual[i,j], dx[2])
                        plane_integral += row_integral
                    plane_integral = self.simpson_integrate_1d(plane_integral.unsqueeze(0), dx[1])
                    integral += plane_integral
                integral = self.simpson_integrate_1d(integral.unsqueeze(0), dx[0])
            else:  # trapezoidal or riemann
                integral = torch.sum(squared_residual) * dx[0] * dx[1] * dx[2]

        return integral

    def __call__(self, f, f_pred):
        """
        Compute the variational loss with improved precision.

        Args:
            f (torch.Tensor): Exact solution or source term.
            f_pred (torch.Tensor): Predicted solution or source term.

        Returns:
            torch.Tensor: Variational loss.
        """
        # Input validation
        if f.shape != f_pred.shape:
            raise ValueError(f"Shape mismatch: f {f.shape} vs f_pred {f_pred.shape}")
        
        # Compute residual
        residual = self.compute_residual(f, f_pred)
        
        # Integrate
        integral = self.integrate(residual, self.dx)
        
        # Apply reduction
        if self.reduction == "mean":
            return integral / f.numel()
        elif self.reduction == "sum":
            return integral
        else:  # "none"
            return integral.unsqueeze(0)
    