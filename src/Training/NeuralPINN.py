import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from scipy.integrate import dblquad


class Net(nn.Module):
    def __init__(self,N_input ,N_output , N_Hidden ,N_layers):
        super(Net,self).__init__()
        #activation = SinActivation()
        activation = nn.Tanh()
        self.f1 = nn.Sequential(*[nn.Linear(N_input ,N_Hidden) , activation])
        self.f2 = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_Hidden , N_Hidden), activation]) for _ in range(N_layers)])
        self.f3 = nn.Sequential(*[nn.Linear(N_Hidden,N_output)])
    def forward(self , x) : 
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return x
    

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

    

"""class Net(nn.Module):
    def __init__(self, N_input, N_output, N_Hidden, N_layers):
        super(Net, self).__init__()
        activation = nn.ReLU()
        
        # CNN layer
        self.conv = nn.Sequential(
            nn.Conv2d(N_input, N_Hidden, kernel_size=3, padding=1),
            activation
        )
        
        # Calculate flattened size
        # Assuming 32x32 input size
        
        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(N_Hidden, N_output)
        )

    def forward(self, x):
        # Reshape if input is not 4D
        if len(x.shape) < 4:
            x = x.unsqueeze(1)  # Add channel dimension
            
        # CNN
        x = self.conv(x)
        # Flatten
        x = x.reshape(32 ,32 , 100)
        
        # MLP
        x = self.mlp(x)
        return x"""

class Integration2DGNN(MessagePassing):
    def __init__(self, in_channels=2, hidden_channels=64):
        super().__init__(aggr='add')
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Softplus()
        )
    
    def forward(self, x, edge_index):
        node_features = self.node_mlp(x)
        hidden = self.propagate(edge_index, x=node_features)
        weights = self.weight_mlp(hidden)
        return weights
    
    def integrate(self, x, edge_index, f_values):
        weights = self.forward(x, edge_index)
        return torch.sum(weights * f_values)
    
class PDE_Loss(object):

    def __init__(self, loss = F.mse_loss):
        super(PDE_Loss,self).__init__()
        self.loss = loss

    def rel(f , f_pred):
        return self.loss(f.squeeze() , f_pred.squeeze())
    
    def abs(f , f_pred):
        return self.loss(f.squeeze() , f_pred.squeeze())
    
    def __call__(self, f , f_pred):
        return self.rel(f , f_pred)
    

class BC_Loss(object):

    def __init__(self):
        super(BC_Loss,self).__init__()


    def rel(u_pred  ,BC):
        u = u_pred.squeeze()
        loss = F.mse_loss(u[:, 0],BC[0]) + F.mse_loss(u[:, -1],BC[1]) + F.mse_loss(u[0,:],BC[2]) + F.mse_loss(u[-1,:],BC[3])
        
        return loss
    
    def abs(u_pred , BC):

        u = u_pred.squeeze()
        loss = F.mse_loss(u[:, 0],BC[0]) + F.mse_loss(u[:, -1],BC[1]) + F.mse_loss(u[0,:],BC[2]) + F.mse_loss(u[-1,:],BC[3])
        
        return loss
    
    def __call__(self, u_pred , BC):
        return self.rel(u_pred , BC)
        

class PINO:

    def __init__(self,model = None , optimizer = None , schedular = None, device = 'cpu'):
        self.device = device
        
        if model is None:
            self.model = Net(1,1, 100, 2).to(device)
        else:
            self.model = model.to(device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        else:
            self.optimizer = optimizer

        if schedular is None:
            self.schedular = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        else:
            self.schedular = schedular
        


    def derivation(self ,u , x ,  order = 2 ):

        for _ in range(order):
            u = torch.autograd.grad(u.sum(),x,create_graph=True , retain_graph= True , allow_unused= True)[0]

        return u
    
    def Poisson_operator(self , u , x , y):

        u_xx = self.derivation(u , x , 2)   
        u_yy = self.derivation(u , y , 2)
        
        return  u_xx + u_yy
    
    def compute_BC_loss(self, u , u_BC):

        BC_loss = F.mse_loss(u, u_BC)
        return BC_loss
    
    def compute_pde_loss(self,u, x, y , f , loss_function = F.mse_loss , decoder = None):
        
        pde = self.Poisson_operator(u, x, y)
        if decoder is not None:
            f = decoder(f)[: , :, 1]
        pde_loss =  loss_function(pde.squeeze(), f.squeeze())

        return pde_loss

    def train_one_epoch(self , x , y , f_exact , BC , u_decoder = None,f_decoder = None, P = None , loss_function = F.mse_loss):

        if P is None:
            loss = self.train_one_batch(x, y, f_exact, BC, u_decoder ,f_decoder  , 1 , loss_function)
        else:
            for p in P:
                loss = self.train_one_batch(x, y, f_exact, BC, u_decoder ,f_decoder , p , loss_function)


        return loss.item()

    def train_one_batch(self , x , y , f_exact , BC , u_decoder = None , f_decoder = None , p = 1 , loss_function = F.mse_loss):
        
        self.optimizer.zero_grad()
        f = f_exact(x ,y ,p)
        u = self.model(f)
        if u_decoder is not None:
            u = u_decoder(u)[: , :, 1].unsqueeze(-1)

        self.pde_loss = self.compute_pde_loss(u, x, y, f , loss_function , f_decoder)
        
        self.BC_loss = F.mse_loss(u[:, 0],BC[0]) + F.mse_loss(u[:, -1],BC[1]) + F.mse_loss(u[0,:],BC[2]) + F.mse_loss(u[-1,:],BC[3])
        #self.BC_loss = self.l2loss(u[:, 0],BC[0]) + self.l2loss(u[:, -1],BC[1]) + self.l2loss(u[:,0],BC[2]) + self.l2loss(u[:,-1],BC[3])
        
        loss = self.pde_loss #+ self.BC_loss
        loss.backward(retain_graph=True)    
        self.optimizer.step()
        return loss
    
    def train(self,epochs, x, y, f_exact, u_BC, u_decoder = None , f_decoder = None , P = None , loss_function = F.mse_loss):
        for epoch in range(epochs):
            loss = self.train_one_epoch( x, y, f_exact, u_BC, u_decoder, f_decoder , P , loss_function)
            if epoch % 100 == 0:
                print(f'Epoch {epoch} Loss {loss} , pde loss {self.pde_loss} , BC loss {self.BC_loss}')

    def predict(self, x , y , f_exact):
        
        f = f_exact(x,y,1)
        return self.model(f).detach().cpu().numpy()