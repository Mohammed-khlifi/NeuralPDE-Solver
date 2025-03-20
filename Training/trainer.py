import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import grad
from tqdm import tqdm
from numpy.polynomial.legendre import legval
import wandb
import seaborn as sns
from Models.models import PINN_Net, CustomPINN
from Operators.Diff_Op import pdeOperator
from Operators.Bound_Op import BoundaryCondition, BoundaryLoss, BoundaryType, BoundaryLocation


class Trainer:
    def __init__(
        self,
        coords,
        boundary_conditions,
        pde_configurations,
        watch=False,
        name=None,
        model = None,
        input_size=2,
        output_size=1,
        hidden=100,
        layers=5,
        lr=0.01
    ):
        """
        Physics-Informed Neural Network (PINN) class for solving PDEs.

        Parameters
        ----------
        coords : list of torch.Tensor
            Coordinate tensors (e.g. for 1D: [x], for 2D: [x,y], etc.)
        f : callable
            Forcing (source) function of the PDE.
        boundary_conditions : list of dict
            List of boundary conditions, each dict must have keys:
            {'type': str, 'location': str, 'value': torch.Tensor}.
        pde_configurations : callable
            PDE pde_configurations that takes u_pred and coords and returns the PDE residual.
        watch : bool, optional
            If True, log training with wandb.
        name : str, optional
            Name for the wandb run (if watch=True).
        u_exact : callable, optional
            Exact solution for reference or error computations.
        input_size : int
            Dimensionality of input to the neural net (e.g. 1, 2, or 3).
        output_size : int
            Dimensionality of the network output (usually 1).
        hidden : int
            Number of hidden units in each layer.
        layers : int
            Number of hidden layers.
        lr : float
            Learning rate.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.coords = coords
        self.f = pde_configurations.source_function 
        self.boundary_conditions = boundary_conditions
        self.pde_configurations = pde_configurations
        self.operator = pde_configurations.operator
        self.exact_solution = pde_configurations.u_exact
        self.watch = watch
        self.lr = lr

        # Build the model
        if model is None:
            self.model = self.build_model(input_size, output_size, hidden, layers)
        else:
            self.model = model

        # Initialize wandb if needed
        if self.watch:
            run_name = self._construct_run_name(name)
            wandb.init(project="pinn-project", name=run_name, reinit=True)
            wandb.watch(self.model)

        # Learnable weights for adaptive boundary conditions
        self.weights = []
        self.weights_handler()

        # Optimizer and Scheduler
        self.optimizer = torch.optim.Adam(
            [{'params': self.model.parameters()}, {'params': self.weights}], lr=self.lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

    def build_model(self, input_size, output_size, hidden, layers):
        """Build the neural network model."""
        model = PINN_Net(input_size, output_size, hidden, layers).to(self.device)

        return model

    def _construct_run_name(self, name):
        """Construct a run name for wandb logging based on dimension and user-specified name."""
        dim = len(self.coords)
        if dim == 1:
            prefix = "1D"
        elif dim == 2:
            prefix = "2D"
        else:
            prefix = "3D"
        return prefix + (name if name else "")
    
    def weights_handler(self):
        """Handle adaptive weights."""
        if self.pde_configurations.trainable:
            weight = self.pde_configurations.weight
            weight.requires_grad = True
            self.weights.append(weight.to(self.device))

        for boundarie in self.boundary_conditions:
            if boundarie.trainable :
                weight = boundarie.weight
                weight.requires_grad = True
                self.weights.append(weight.to(self.device))
            else:
                pass


    def compute_loss(self, u_pred, coords, f, loss_function=None):
        """Compute the total loss including PDE and boundary conditions."""
        
        # PDE residual
        f_pred = self.operator(u_pred, *coords).squeeze()
        f = f.squeeze()
        pde_loss = loss_function(f_pred, f) if loss_function else F.mse_loss(f_pred, f)
        self.error = torch.abs(f_pred - f)  
 
        # Boundary conditions
        boundary_loss = BoundaryLoss()
        if self.pde_configurations.trainable: 
            trainable_idx = 1  # Start from 1 since weights[0] is for something else
            self.pde_configurations.weight = self.weights[0]
        else:
            trainable_idx = 0
        for i in range(len(self.boundary_conditions)):
            if self.boundary_conditions[i].trainable:
                self.boundary_conditions[i].weight = self.weights[trainable_idx]
                trainable_idx += 1 
             
        boundary_losses = [boundary_loss(u_pred, bc , coords) for bc in self.boundary_conditions]

        # Weight function (by default: exponential(-weight))
        weight_function = self.pde_configurations.weight_function  

        # Total loss
        loss = sum(boundary_losses[i] for i in range(len(boundary_losses)))
        total_loss = pde_loss*weight_function(self.pde_configurations.weight) + loss
            

        # Logging to wandb
        if self.watch:
            self._log_to_wandb(total_loss, pde_loss, boundary_losses)

        return total_loss

    def _log_to_wandb(self, total_loss, pde_loss, boundary_losses):
        """Log metrics to wandb."""
        wandb.log({"loss": total_loss.item(), "pde_loss": pde_loss.item()})
        for i, bc_loss in enumerate(boundary_losses):
            wandb.log({f"boundary_loss_{i}": bc_loss.item()})
        for i, weight in enumerate(self.weights):
            wandb.log({f"weight_{i}": torch.exp(-weight)})


    def train_one_epoch(self, epoch, inputs , coords, rate=100, loss_function=None):
        self.optimizer.zero_grad()
        
        # Forward pass
        u_pred = self.model(inputs).squeeze()
        
        # Compute exact solution if available
        u_exact = self.exact_solution(*coords) if self.exact_solution else None
        
        # Compute forcing function
        f_values = self.f(*coords)

        # Compute loss
        loss = self.compute_loss(u_pred, coords, f_values, loss_function)

        # Adaptive collocation
        if self.pde_configurations.adaptive_nodes > 0 and (epoch + 1) % rate == 0:
            # Update collocation points
            coords = self.update_collocation_points(coords, u_pred, u_exact)
            inputs = torch.cat([c.unsqueeze(-1) for c in coords], dim=-1)

        # Backward pass
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.scheduler.step()

        return loss , coords


    def train(self, epochs=500, rate=100, loss_function=None):
        """Train the PINN model."""
        
        print("Training the model...")
        progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")
        coords = self.coords
        #inputs = torch.cat([c.unsqueeze(-1) for c in coords], dim=-1)
        for epoch in progress_bar:
            
            # Train for one epoch
            inputs = torch.cat([c.unsqueeze(-1) for c in coords], dim=-1)
            loss , coords = self.train_one_epoch(epoch, inputs , coords, rate, loss_function)

            # Update progress bar
            mse = self.test() if self.exact_solution else None
            postfix = {"Loss": f"{loss.item():.4e}"}
            if mse is not None:
                postfix["MSE"] = f"{mse.item():.4e}"
                if self.watch:
                    wandb.log({"MSE": mse.item()})
            progress_bar.set_postfix(postfix)
        self.coords = coords
        if self.watch:
            wandb.finish()

        final_mse = mse.item() if mse is not None else 0
        return final_mse, loss.item()

    def adaptive_collocation_points(self, coords, u_pred, u_exact, num_samples=100 , noise_level = 0.0001):
        """Select new collocation points based on residual distribution."""
        fake_inputs = torch.cat([c.unsqueeze(-1) for c in coords], dim=-1)
        u_pred = self.model(fake_inputs).squeeze()
        residuals = torch.abs(self.operator(u_pred, *coords).squeeze() - self.f(*coords).squeeze())
        p = (residuals**2) / torch.sum(residuals**2)
        indices = torch.multinomial(p.flatten(), num_samples=num_samples, replacement=True)
        new_coords = []

        for c in coords:
            new_coord = c.flatten()[indices]
            new_coords.append(new_coord.squeeze())
            
        return new_coords

    def update_collocation_points(self, coords, outputs, u_exact):
        """
        Update collocation points for adaptive sampling.

        Args:
            coords (list of torch.Tensor): Current collocation points (1D, 2D, or 3D).
            outputs (torch.Tensor): Model predictions at current collocation points.
            u_exact (torch.Tensor): Exact solution values (if available).

        Returns:
            list of torch.Tensor: Updated collocation points with adaptive sampling.
        """

        num_samples = self.pde_configurations.adaptive_nodes

        # Handle 1D case
        if len(coords) == 1:
            a = coords[0].min().item()
            b = coords[0].max().item()
            fake_x = torch.linspace(a, b, 1000, device=self.device, requires_grad=True)
            adaptive_x = self.adaptive_collocation_points([fake_x], outputs, u_exact, num_samples=num_samples)[0]
            
            # Combine old and new collocation points
            x = torch.cat([coords[0], adaptive_x])
            coords[0] = torch.sort(x)[0]

        # Handle 2D case
        elif len(coords) == 2:
            a = coords[0].min().item()
            b = coords[0].max().item()
            c = coords[1].min().item()
            d = coords[1].max().item()
            fake_x = torch.linspace(a, b, 100, device=self.device)
            fake_y = torch.linspace(c, d, 100, device=self.device)
            fake_x, fake_y = torch.meshgrid(fake_x, fake_y, indexing="ij")

            fake_x.requires_grad = fake_y.requires_grad = True

            adaptive_x, adaptive_y = self.adaptive_collocation_points([fake_x, fake_y], outputs, u_exact, num_samples=num_samples)
            x, y = coords[0], coords[1]
            
            # Compute the shape for adaptive sampling
            n_shape = int(num_samples ** 0.5 + 0.5)
            adaptive_x = adaptive_x.reshape(n_shape, n_shape)
            adaptive_y = adaptive_y.reshape(n_shape, n_shape)

            # Merge and sort collocation points
            x = torch.cat([x[:, 0], adaptive_x[:, 0]])
            y = torch.cat([y[0, :], adaptive_y[0, :]])
            x, y = torch.sort(x)[0], torch.sort(y)[0]
            coords = torch.meshgrid(x, y, indexing="ij")

        # Handle 3D case
        elif len(coords) == 3:
            a = coords[0].min().item()
            b = coords[0].max().item()
            c = coords[1].min().item()
            d = coords[1].max().item()
            e = coords[2].min().item()
            f = coords[2].max().item()
            
            fake_x = torch.linspace(a, b, 50, device=self.device)
            fake_y = torch.linspace(c, d, 50, device=self.device)
            fake_z = torch.linspace(e, f, 50, device=self.device)
            fake_x, fake_y, fake_z = torch.meshgrid(fake_x, fake_y, fake_z, indexing="ij")

            fake_x.requires_grad = fake_y.requires_grad = fake_z.requires_grad = True

            adaptive_x, adaptive_y, adaptive_z = self.adaptive_collocation_points([fake_x, fake_y, fake_z], outputs, u_exact, num_samples=num_samples)
            x, y, z = coords[0], coords[1], coords[2]

            # Compute the shape for adaptive sampling
            n_shape = int(num_samples ** (1/3) + 0.5)
            adaptive_x = adaptive_x.reshape(n_shape, n_shape, n_shape)
            adaptive_y = adaptive_y.reshape(n_shape, n_shape, n_shape)
            adaptive_z = adaptive_z.reshape(n_shape, n_shape, n_shape)

            # Merge and sort collocation points
            x = torch.cat([x[:, 0, 0], adaptive_x[:, 0, 0]])
            y = torch.cat([y[0, :, 0], adaptive_y[0, :, 0]])
            z = torch.cat([z[0, 0, :], adaptive_z[0, 0, :]])
            x, y, z = torch.sort(x)[0], torch.sort(y)[0], torch.sort(z)[0]
            coords = torch.meshgrid(x, y, z, indexing="ij")

        # Update exact solution if available
        if self.exact_solution is not None:
            u_exact = self.exact_solution(*coords)

        return coords

    
    def visualize_distribution(self , coords , error ):
        """
        Visualize distribution of vector elements
        Args:
            coords: List of coordinate tensors
            selected_indices: Indices of selected points
            original_shape: Original shape of the domain
        """
        fig = plt.figure(figsize=(15, 5))
        
        # 1. Histogram of values
        plt.subplot(131)
        for i, coord in enumerate(['x']):
            sns.histplot(coords[i].flatten().detach().numpy(), label=coord, alpha=0.3)
        plt.title('Distribution of Coordinates')
        plt.legend()
        # plt of error
        plt.subplot(132)
        plt.plot(error.detach().numpy()  , label = "Error")

    def plot_solution(self, coords, u_exact):
        """Visualize the solution for 1D, 2D, or 3D PDEs."""
        inputs = torch.cat([c.unsqueeze(-1) for c in coords], dim=-1)
        u_pred = self.model(inputs).cpu().detach().numpy().squeeze()

        dim = len(coords)
        if dim == 1:
            self._plot_1D(coords, u_pred, u_exact)
        elif dim == 2:
            self._plot_2D(coords, u_pred, u_exact)
        elif dim == 3:
            self._plot_3D(coords, u_pred, u_exact)
        else:
            raise ValueError("Unsupported dimension for plotting.")

    def _plot_1D(self, coords, u_pred, u_exact):
        plt.plot(coords[0].cpu().detach().numpy().squeeze(), u_pred, label="Prediction", linestyle="--")
        if u_exact is not None:
            plt.plot(coords[0].cpu().detach().numpy().squeeze(),
                     u_exact.cpu().detach().numpy().squeeze(), label="Exact")
        plt.legend()
        plt.show()

    def _plot_2D(self, coords, u_pred, u_exact):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        X = coords[0].cpu().detach().numpy()
        Y = coords[1].cpu().detach().numpy()
        cf1 = axes[0].contourf(X, Y, u_pred, levels=100, cmap="jet")
        axes[0].set_title("Prediction")
        fig.colorbar(cf1, ax=axes[0])

        if u_exact is not None:
            cf2 = axes[1].contourf(X, Y, u_exact.cpu().detach().numpy(), levels=100, cmap="jet")
            axes[1].set_title("Exact")
            fig.colorbar(cf2, ax=axes[1])

        plt.show()

    def _plot_3D(self, coords, u_pred, u_exact):
        x_test, y_test, z_test = [c.cpu().detach().numpy().squeeze() for c in coords]
        slice_idx = u_pred.shape[2] // 2
        x_slice = x_test[:, :, slice_idx]
        y_slice = y_test[:, :, slice_idx]
        u_slice = u_pred[:, :, slice_idx]
        u_exact_slice = u_exact.cpu().detach().numpy()[:, :, slice_idx] if u_exact is not None else None

        fig = plt.figure(figsize=(20, 7))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(x_slice, y_slice, u_slice, cmap="viridis")
        ax1.set_title("Predicted Solution (z=mid)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("u_pred")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=12)

        if u_exact_slice is not None:
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            surf2 = ax2.plot_surface(x_slice, y_slice, u_exact_slice, cmap="viridis")
            ax2.set_title("Exact Solution (z=mid)")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("u_exact")
            fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=12)

        plt.tight_layout()
        plt.show()

    def test(self):
        """
        Test the model on a fixed grid and compute MSE with the exact solution if available.
        """
        dimensions = len(self.coords)
        inputs = torch.cat([c.unsqueeze(-1) for c in self.coords], dim=-1)

        u_test = self.model(inputs)
        mse = F.mse_loss(u_test.squeeze(), self.exact_solution(*self.coords).squeeze())
        return mse
    
    def get_coords(self):
        return self.coords
    
    def get_risidual(self):
        return self.error
