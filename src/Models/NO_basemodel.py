from .basemodel import Basemodel
from .models import PINN_Net, CustomPINN
from src.Operators.Bound_Op import BoundaryCondition, BoundaryLocation, BoundaryType    
from src import pdeOperator , OperatorConfig
import torch
import torch.nn as nn
import typing as tp
from src import FNO
from src.neuraloperator.neuralop import Trainer
from src.neuraloperator.neuralop import LpLoss, H1Loss
import torch.nn.functional as F

class NO_basemodel(Basemodel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        


        """self.model = FNO(n_modes=(16, 16),
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            projection_channel_ratio=2)"""
        
        #self.model = quickMLP(1, 1,  self.param['N_hidden'], self.param['N_layers'])
        self.model = quickCNN(1, 1, 16, 3)
        self.learning_rate = self.param['lr']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.1)
        self.trainer = Trainer(model= self.model, n_epochs= self.param['epochs'],
                  device='cpu',
                  wandb_log=True,
                  eval_interval=3,
                  use_distributed=False,
                  verbose=True) 

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
                #pde_loss = self.calculate_pde_loss(x, y_pred)
                #total_loss = mse_loss + self.param['weight'] * pde_loss
                
                # Backward pass
                mse_loss.backward(retain_graph=True)
                self.optimizer.step()
                
                
                epoch_loss += mse_loss.item()
            
            # Validation phase
            val_loss = self.validate(test_loaders)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                
            # Save best model
            """if val_loss < best_loss:
                best_loss = val_loss
                if self.param.get('save_version'):
                    #self.save(f"model_epoch_{epoch}.pt")"""
            
            # Log metrics
            #train_losses.append(epoch_loss/len(train_loader))
            #val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {epoch_loss/len(train_loader):.4e}, "
                    f"Val Loss = {val_loss:.4e}, H1 Loss = {h1_loss:.4e}")
        if self.param['save_version']: 
            #print(self.param['save_version'])
            self.save_version(self.model, {"train loss": epoch_loss/len(train_loader), "val loss": val_loss})
        return epoch_loss/len(train_loader), val_loss

    def validate(self, test_loaders):
        """Validation step"""
        self.model.eval()
        total_loss = 0.0

        
        with torch.no_grad():
            for test_loader in test_loaders.values():
                for batch in test_loader:
                    x = batch['x']
                    y_true = batch['y']
                    y_pred = self.model(x)
                    
                    loss = F.mse_loss(y_pred.flatten(), y_true.flatten())
                    total_loss += loss.item()
        
        return total_loss / len(test_loaders)

    def calculate_pde_loss(self, x, y_pred):
        """Calculate PDE residual loss"""
        return self.operator(y_pred, x)


            
class quickMLP(nn.Module):
    def __init__(self,N_input ,N_output , N_Hidden ,N_layers):
        super(quickMLP,self).__init__()
        activation = nn.Tanh()
        #activation = SinActivation()
        self.f1 = nn.Sequential(*[nn.Linear(N_input ,N_Hidden) , activation])
        self.f2 = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_Hidden , N_Hidden), activation]) for _ in range(N_layers)])
        self.f3 = nn.Sequential(*[nn.Linear(N_Hidden,N_output)])
    def forward(self , x) :
        x = x.unsqueeze(-1)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = x.squeeze()
        return x

class quickCNN(nn.Module):
    def __init__(self, N_input, N_output, N_Hidden, N_layers, kernel_size=3, num_filters=16):
        """
        CNN equivalent of quickMLP with 1D convolutions instead of linear layers.

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
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding=1),
            activation
        )

        # Hidden Convolutional Layers (Mimics Hidden Fully Connected Layers)
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
