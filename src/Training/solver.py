from src.Training.trainer import Trainer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import grad
from tqdm import tqdm
from numpy.polynomial.legendre import legval
import wandb

class Solver:
    """
    A Solver class for solving PDEs in 1D, 2D, or 3D using neural networks.
    This class supports training a neural network-based model to approximate solutions
    and adaptively refine the solution over time.
    """
    def __init__(
        self,
        input_size=None,
        output_size=None,
        hidden_size=100,
        num_layers=1,
        learning_rate=0.01,
        device=None,
        wandb_logs=False,
        name=None
    ):
        """
        Initialize the Solver instance with default settings.
        
        Parameters:
        ----------
        input_size : int, optional
            Number of input features (e.g., spatial dimensions).
        output_size : int, optional
            Number of output features (e.g., the PDE solution).
        hidden_size : int, default=100
            Number of neurons in each hidden layer.
        num_layers : int, default=1
            Number of hidden layers.
        learning_rate : float, default=0.01
            Learning rate for the optimizer.
        device : str, optional
            Device to run the computations on ('cuda' or 'cpu').
        wandb_logs : bool, default=False
            If True, enable logging with Weights & Biases (wandb).
        name : str, optional
            Name for logging or identification purposes.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_logs = wandb_logs
        self.name = name

    def solve(
        self,
        coords=None,
        pde_configurations=None,
        boundary_conditions=None,
        epochs=1001,
        dimensions=None,
        plot=True,
        num_test_points=10,
    ):
        """
        Solve the PDE for a given domain and configuration.
        
        Parameters:
        ----------
        coords : list
            Domain points for the PDE (1D: [x], 2D: [x, y], 3D: [x, y, z]).
        pde_configurations : object
            Configuration containing the PDE operator, source function, and exact solution.
        boundary_conditions : list
            Boundary conditions to enforce during training.
        epochs : int, default=1001
            Number of training epochs.
        dimensions : int
            Dimensionality of the problem (1, 2, or 3).
        plot : bool, default=True
            Whether to plot the final solution for 3D problems.
        num_test_points : int, default=10
            Number of test points for plotting or evaluation.

        Returns:
        -------
        Trained model, loss, residuals, and coordinates.
        """
        # Select appropriate solver based on the problem dimension
        if dimensions == 1:
            return self._solve_1D(coords, pde_configurations, boundary_conditions, epochs)
        elif dimensions == 2:
            return self._solve_2D(coords, pde_configurations, boundary_conditions, epochs)
        elif dimensions == 3:
            return self._solve_3D(coords, pde_configurations, boundary_conditions, epochs, plot, num_test_points)
        else:
            raise ValueError(f"Unsupported dimensions: {dimensions}")

    def _solve_1D(self, coords, pde_configurations, boundary_conditions, epochs):
        """
        Solve a 1D PDE using neural networks.
        """
        x = coords[0]
        x = torch.sort(x)[0]
        x.requires_grad = True  # Enable gradient tracking for automatic differentiation

        trainer = Trainer(
            [x], boundary_conditions, pde_configurations,
            input_size=1, output_size=1,
            hidden=self.hidden_size, layers=self.num_layers,
            watch=self.wandb_logs, name=self.name, lr=self.learning_rate
        )

        mse, loss = trainer.train(
            epochs=epochs, rate=pde_configurations.update_rate, loss_function=pde_configurations.pde_loss
        )
        return trainer.model, mse, loss, trainer.get_coords(), trainer.get_risidual()

    def _solve_2D(self, coords, pde_configurations, boundary_conditions, epochs):
        """
        Solve a 2D PDE using neural networks.
        """
        if len(coords) != 2:
            raise ValueError("2D domain must have exactly 2 coordinate sets.")

        x, y = coords
        x, y = torch.sort(x)[0], torch.sort(y)[0]
        x.requires_grad = True
        y.requires_grad = True
        x, y = torch.meshgrid(x, y)

        trainer = Trainer(
            [x, y], boundary_conditions, pde_configurations,
            input_size=2, output_size=1,
            hidden=self.hidden_size, layers=self.num_layers,
            watch=self.wandb_logs, name=self.name, lr=self.learning_rate
        )

        mse, loss = trainer.train(
            epochs=epochs, rate=pde_configurations.update_rate, loss_function=pde_configurations.pde_loss
        )
        return trainer.model, mse, loss, trainer.get_coords(), trainer.get_risidual()

    def _solve_3D(self, coords, pde_configurations, boundary_conditions, epochs, plot, num_test_points):
        """
        Solve a 3D PDE using neural networks.
        """
        if len(coords) != 3:
            raise ValueError("3D domain must have exactly 3 coordinate sets.")

        x, y, z = coords
        x, y, z = torch.sort(x)[0], torch.sort(y)[0], torch.sort(z)[0]
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True
        x, y, z = torch.meshgrid(x, y, z)

        trainer = Trainer(
            [x, y, z], boundary_conditions, pde_configurations,
            input_size=3, output_size=1,
            hidden=self.hidden_size, layers=self.num_layers,
            watch=self.wandb_logs, name=self.name, lr=self.learning_rate
        )

        mse, loss = trainer.train(
            epochs=epochs, rate=pde_configurations.update_rate, loss_function=pde_configurations.pde_loss
        )

        if plot:
            u_exact = pde_configurations.u_exact
            """a, b = x[0].item(), x[-1].item()
            c, d = y[0].item(), y[-1].item()
            e, f_ = z[0].item(), z[-1].item()"""

            # Generate test points
            x_test = torch.linspace(-1, 1, num_test_points, requires_grad=True).to(self.device)
            y_test = torch.linspace(-1, 1, num_test_points, requires_grad=True).to(self.device)
            z_test = torch.linspace(-1, 1, num_test_points, requires_grad=True).to(self.device)
            x_test, y_test, z_test = torch.sort(x_test)[0], torch.sort(y_test)[0], torch.sort(z_test)[0]
            x_test, y_test, z_test = torch.meshgrid(x_test, y_test, z_test)

            # Plot the solution
            trainer.plot_solution([x_test, y_test, z_test], u_exact(x_test, y_test, z_test))

        return trainer.model, mse, loss, trainer.get_coords(), trainer.get_risidual()

"""    def Benchmark(
        self,
        coords = None,

        epochs=1001, 
        dimensions=None, 
        update_rate=250,  
        plot=False, 
        num_test_points=10,

    ):
        
        Benchmark various PINN configurations (e.g., with and without adaptive weighting/collocation,
        variational loss, etc.) across different PDE dimensions.

        Parameters
        ----------
        domaine : list or tuple
            Domain definition for the PDE.
        operator : callable, optional
            PDE operator function.
        source_function : callable, optional
            Source term of the PDE.
        u_exact : callable, optional
            Exact solution for reference.
        epochs : int, default=1001
            Number of training epochs.
        dimensions : int, optional
            PDE dimensionality (1, 2, or 3).
        num_points : int, default=50
            Number of collocation points.
        rate : int, default=250
            Rate for adaptive strategies.
        hidden : int, default=100
            Number of hidden units per layer.
        layers : int, default=1
            Number of hidden layers.
        watch : bool, default=False
            If True, enable logging (e.g., wandb).
        lr : float, default=0.01
            Learning rate.
        plot : bool, default=True
            If True, plot solutions after training.
        n_test : int, default=10
            Number of test points for evaluation/plotting.

        Returns
        -------
        (Models_names, Models) : (list of str, list of models)
            Models_names is a list of model configuration names.
            Models is a list of the trained models corresponding to each configuration.
        
        if dimensions is None:
            raise ValueError("dimensions must be specified (1, 2, or 3).")

        # Define model configurations based on dimensions
        # Each entry: (name, adaptive_collocation)
        # The indexes are for explanation; the sets differ by dimension.
        if dimensions == 1:
            model_configs = [
                ("PINN",       False, False),
                ("AC-PINN",    False, True),
                ("AW-PINN",    True,  False),
                ("ACAW-PINN",  True,  True),
            ]
            dim_str = "1D"
        elif dimensions == 2:
            model_configs = [
                ("PINN",      False, False),
                ("AC-PINN",   False, True),
                ("AW-PINN",   True,  False),
                ("ACAW-PINN", True,  True),
            ]
            dim_str = "2D"
        elif dimensions == 3:
            # If 3D supports V-PINN or others, adjust accordingly.
            # Assuming similar to 1D with V-PINN variants:
            model_configs = [
                ("PINN",   False),
                ("AC-PINN",    True),
                ("AW-PINN",    False),
                ("ACAW-PINN", True),
                # Add V-PINN variants if needed:
                # ("V-PINN",     False, False, True),
                # ("AW-VPINN",   True,  False, True),
            ]
            dim_str = "3D"
        else:
            raise ValueError(f"Unsupported dimensions: {dimensions}")

        print(f"Starting benchmark for {dim_str} PDE models.")
        Models_names = [cfg[0] for cfg in model_configs]
        Models = []
        MSE = []
        LOSS = []

        for name, ac in model_configs:
            print(f"Training {name} ...")
            model ,mse , loss = self.solve(
                coords=coords,  
                pde_configurations=pde_configurations,
                epochs=epochs,
                dimensions=dimensions,
                update_rate=update_rate,
                plot=plot,
                num_test_points=num_test_points,
            )
            Models.append(model)
            MSE.append(mse)
            LOSS.append(loss)
            print(f"Finished training {name}.\n{'-'*40}")

        print(f"Benchmark for {dim_str} completed. Trained models: {Models_names}")
        return Models_names, Models"""
