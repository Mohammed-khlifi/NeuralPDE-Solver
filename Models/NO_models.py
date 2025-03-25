from .NO_basemodel import NO_basemodel
from neuraloperator.neuralop.models import FNO , UNO ,GINO, UQNO , FNOGNO, TFNO
import torch
import torch.nn as nn





class CNN_model(NO_basemodel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = quickCNN(1, 1, 3, 4)
        self.learning_rate = self.param['lr']
        
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        



class FNO_model(NO_basemodel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
        modes = self.param['modes']
        self.model = FNO(n_modes=(modes, modes),
            in_channels=self.param['in_channels'],
            out_channels=self.param['out_channels'],
            hidden_channels=self.param['hidden_channels'],
            projection_channel_ratio=self.param['projection_channel_ratio'])
        
        self.learning_rate = self.param['lr']
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

class TFNO_model(NO_basemodel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        modes = self.param['modes']
        self.model = TFNO(n_modes=(modes, modes), hidden_channels=self.param['hidden_channels'],
                in_channels=self.param['in_channels'], out_channels=self.param['out_channels'],
                factorization='tucker',
                implementation='factorized',
                rank=0.05)
        
        self.learning_rate = self.param['lr']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.1)

class UNO_model(NO_basemodel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        self.model = UNO(in_channels=1,
            out_channels=1,
            hidden_channels=64,
            projection_channels=64,
            uno_out_channels=[32,64,64,64,32],
            uno_n_modes=[[16,16],[8,8],[8,8],[8,8],[16,16]],
            uno_scalings=[[1.0,1.0],[0.5,0.5],[1,1],[2,2],[1,1]],
            horizontal_skips_map=None,
            channel_mlp_skip="linear",
            n_layers = 5,
            domain_padding=0.2)
        
        
        self.learning_rate = self.param['lr']
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()


class PINO_model(NO_basemodel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        self.model = FNO(n_modes=(16, 16),
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            projection_channel_ratio=2)
        self.learning_rate = self.param['lr']
        self.weights = [torch.tensor(0.0), torch.tensor(10.0)]
        #kdkdkdk
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.learning_rate},
            {'params': self.weights, 'lr': self.learning_rate}
        ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.1)

    def fit(self, train_loader, test_loaders):
        """Custom training loop"""
        train_loss = H1Loss(d=2)
        l2loss = LpLoss(d=2 ,p=2)
        
        # Initialize tracking
        best_loss = float('inf')
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(self.param['epochs']):
            # Train phase
            self.model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                x = batch['x']
                y_true = batch['y']
                y_pred = self.model(x)
                
                # Calculate losses
                mse_loss = F.mse_loss(y_pred.flatten(), y_true.flatten())
                h1_loss = train_loss(y_pred.reshape(y_true.shape), y_true)
                pde_loss = self.calculate_pde_loss(y_true, y_pred)
                total_loss = mse_loss * torch.exp(-self.weights[0]) + pde_loss * torch.exp(-self.weights[1]) + torch.abs(self.weights[0]) + torch.abs(self.weights[1])

                
                # Backward pass
                total_loss.backward(retain_graph=True)
                self.optimizer.step()
                
                
                epoch_loss += mse_loss.item()
            
            # Validation phase
            val_loss = self.validate(test_loaders)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {epoch_loss/len(train_loader):.4e}, "
                    f"Val Loss = {val_loss:.4e}, H1 Loss = {h1_loss:.4e} pde_loss = {pde_loss:.4e}")
        
        if self.param['save_version']: 
            self.save_version(self.model, {"train loss": epoch_loss/len(train_loader), "val loss": val_loss})

        return epoch_loss/len(train_loader), val_loss
                         
    def calculate_pde_loss(self, u_exact , u_pred):
        """Calculate PDE loss for Poisson equation
        """

        fexact_fdm = self.laplacian(u_exact.squeeze())
        fapprox_fdm = self.laplacian(u_pred)
        pde_loss = F.mse_loss(fexact_fdm, fapprox_fdm)
        return pde_loss
        
    
    def laplacian(self, y_pred):
        """Calculate Laplacian using FVM"""
        y_pred = y_pred.squeeze()
        batch_size, nx , ny = y_pred.shape
        dx = 1 / nx
        dy = 1 / ny
        du_dx = (y_pred[:, 2:, 1:-1] - y_pred[:, :-2, 1:-1]) / (2 * dx)
        du_dy = (y_pred[:, 1:-1, 2:] - y_pred[:, 1:-1, :-2]) / (2 * dy)
        #print(du_dx.shape)
        #import sys ; sys.exit()
        # Calculate second derivatives
        d2u_dx2 = (y_pred[:, 2:, 1:-1] - 2*y_pred[:, 1:-1, 1:-1] + y_pred[:, :-2, 1:-1]) / dx**2
        d2u_dy2 = (y_pred[:, 1:-1, 2:] - 2*y_pred[:, 1:-1, 1:-1] + y_pred[:, 1:-1, :-2]) / dy**2

        laplacian = d2u_dx2 + d2u_dy2
        
        return laplacian
        

class quickCNN(nn.Module):
    def __init__(self, N_input, N_output, N_Hidden, N_layers, kernel_size=3, num_filters=16):
        """

        Args:
            N_input (int): Input feature size.
            N_output (int): Output size.
            N_Hidden (int): Number of hidden neurons (used for FC layer after CNN).
            N_layers (int): Number of hidden layers.
            kernel_size (int): Size of convolutional kernel.
            num_filters (int): Number of filters in convolution layers.
        """
        super(quickCNN, self).__init__()
        activation = nn.Tanh()

        # Initial Convolutional layer (Mimics First Linear Layer)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=N_input, out_channels=num_filters, kernel_size=kernel_size, padding=1),
            activation
        )

        # Hidden Convolutional Layers (
        self.conv_hidden = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=1),
                    activation
                ) for _ in range(N_layers)
            ]
        )

        # Fully Connected Layer (To map CNN features to output)
        self.fc = nn.Sequential(
            nn.Linear(num_filters, N_Hidden),  # Flattened CNN output
            activation,
            nn.Linear(N_Hidden, N_output)
        )

    def forward(self, x):
        """
        Forward pass through the CNN model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, N_input)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, N_output)
        """
        # Reshape input to fit CNN (Batch, Channels, Features)
        #x = x.unsqueeze(1)  # Convert (Batch, Features) -> (Batch, Channels=1, Features)

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv_hidden(x)

        # Flatten the CNN output for fully connected layer
        x = x.permute(0, 2, 3, 1)
        
        # Fully connected output mapping
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2)

        return x