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
        #self.model = PINN_Net(16, 16, self.param['N_hidden'], self.param['N_layers'])
        self.model = quickMLP(16*16, 16*16,  self.param['N_hidden'], self.param['N_layers'])
        self.learning_rate = self.param['lr']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.1)
        self.trainer = Trainer(model= self.model, n_epochs= 100,
                  device='cpu',
                  wandb_log=True,
                  eval_interval=3,
                  use_distributed=False,
                  verbose=True) 

    def fit(self, train_loader, test_loaders, epochs=500):
        """Custom training loop"""
        train_loss = H1Loss(d=2)
        l2loss = LpLoss(d=2 ,p=2)
        
        # Initialize tracking
        best_loss = float('inf')
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(epochs):
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
                h1_loss = 0#train_loss(y_pred.flatten(), y_true.flatten())
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
        
        return train_loss, val_loss

    def validate(self, test_loaders):
        """Validation step"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loaders:
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
        x = x.view(x.size(0), -1)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = x.view(x.size(0), 1, 16, 16)
        return x