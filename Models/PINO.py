from .NO_basemodel import NO_basemodel
from neuraloperator.neuralop.models import FNO , UNO ,GINO, UQNO , FNOGNO
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuraloperator.neuralop import LpLoss, H1Loss


class PINO_poisson(NO_basemodel):
    
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

    def fit(self, train_loader, test_loaders ):
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
                total_loss = h1_loss * torch.exp(-self.weights[0]) + pde_loss * torch.exp(-self.weights[1]) + torch.abs(self.weights[0]) + torch.abs(self.weights[1])

                
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
    

class PINO_darcy(NO_basemodel):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        
        self.model = FNO(n_modes=(16, 16),
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            projection_channel_ratio=2)
        self.learning_rate = self.param['lr']
        self.weights = [torch.tensor(0.0), torch.tensor(10.0)]
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
                pde_loss = self.calculate_pde_loss(y_true, y_pred ,x)
                total_loss = h1_loss * torch.exp(-self.weights[0]) + pde_loss * torch.exp(-self.weights[1]) + torch.abs(self.weights[0]) + torch.abs(self.weights[1])

                
                # Backward pass
                total_loss.backward(retain_graph=True)
                self.optimizer.step()
                
                
                epoch_loss += mse_loss.item()
            
            # Validation phase
            val_loss = self.validate(test_loaders)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Train Loss = {epoch_loss/len(train_loader):.4e}, "
                    f"Val Loss = {val_loss:.4e}, H1 Loss = {h1_loss:.4e} pde_loss = {pde_loss:.4e}")
        
        if self.param['save_version']: 
            self.save_version(self.model, {"train loss": epoch_loss/len(train_loader), "val loss": val_loss})

        return epoch_loss/len(train_loader) , val_loss
                         
    def calculate_pde_loss(self, u_exact , u_pred ,x):
        """Calculate PDE loss for Poisson equation
        """

        fexact_fdm = self.darcy_flow(u_exact.squeeze() , x)
        fapprox_fdm = self.darcy_flow(u_pred , x)
        pde_loss = F.mse_loss(fexact_fdm, fapprox_fdm)
        return pde_loss
        
    
    def darcy_flow(self, u , k):
        """Calculate Darcy flow using FVM"""
        
        u = u.squeeze(1)
        k = k.squeeze(1)
        
        # Calculate gradients
        _, nx, ny = u.shape
        dx = 1 / nx
        dy = 1 / ny
        du_dx = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
        du_dy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)
        
        k = k[: , 1:-1 , 1:-1]
        
        dduddx = (du_dx[:, 2:, 1:-1] - du_dx[:, :-2, 1:-1]) / (2 * dx)
        dduddy = (du_dy[:, 1:-1, 2:] - du_dy[:, 1:-1, :-2]) / (2 * dy)
        
        darcy = -(dduddx + dduddy)

        return darcy
