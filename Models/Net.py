import torch
import torch.nn as nn


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
class PINN_Net(nn.Module):
    def __init__(self,N_input ,N_output , N_Hidden ,N_layers):
        super(PINN_Net,self).__init__()
        activation = nn.Tanh()
        #activation = SinActivation()
        self.f1 = nn.Sequential(*[nn.Linear(N_input ,N_Hidden) , activation])
        self.f2 = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_Hidden , N_Hidden), activation]) for _ in range(N_layers)])
        self.f3 = nn.Sequential(*[nn.Linear(N_Hidden,N_output)])
    def forward(self , x) :
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return x
    

    
class CustomPINN(nn.Module):
    def __init__(self, N_input, N_output, N_hidden, activation=nn.Tanh):
        super(CustomPINN, self).__init__()
        self.activation = activation()

        # Input layer
        self.input_layer = nn.Linear(N_input, N_hidden)

        # Explicitly defined hidden layers
        self.hidden1 = nn.Linear(N_hidden, N_hidden)
        self.norm1 = nn.LayerNorm(N_hidden)

        self.hidden2 = nn.Linear(N_hidden, N_hidden)
        self.norm2 = nn.LayerNorm(N_hidden)

        self.hidden3 = nn.Linear(N_hidden, N_hidden)
        self.norm3 = nn.LayerNorm(N_hidden)

        # Output layer
        self.output_layer = nn.Linear(N_hidden, N_output)

    def forward(self, x):
        # Input to first hidden layer
        x = self.activation(self.input_layer(x))

        # Hidden layers with skip connections
        res = x
        x = self.activation(self.hidden1(x))
        x = self.norm1(x)
        x += res

        res = x
        x = self.activation(self.hidden2(x))
        x = self.norm2(x)
        x += res

        res = x
        x = self.activation(self.hidden3(x))
        x = self.norm3(x)
        x += res

        # Output layer
        x = self.output_layer(x)
        return x