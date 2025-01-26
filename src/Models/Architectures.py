import torch 
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,N_input ,N_output , N_Hidden ,N_layers):
        super(Net,self).__init__()
        activation = nn.Tanh()
        self.f1 = nn.Sequential(*[nn.Linear(N_input ,N_Hidden) , activation])
        self.f2 = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_Hidden , N_Hidden), activation]) for _ in range(N_layers)])
        self.f3 = nn.Sequential(*[nn.Linear(N_Hidden,N_output)])
    def forward(self , x) : 
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return x
    
    
class PINN_CNN(nn.Module):
    def __init__(self, N_input,N_output ,width):
        super(PINN_CNN, self).__init__()

        
        self.width = width
        self.padding = 9


        self.fc0 = nn.Linear(N_input, self.width)
        self.fc01 = nn.Linear(self.width, self.width)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, N_output)

    def forward(self, x):
        
        x = self.fc0(x)
        #x = torch.tanh(x)
        x = self.fc01(x)
        #x = torch.tanh(x)
        
        x = x.permute(2, 0, 1).unsqueeze(0)
        x2 = self.w0(x)
        x =  x2 
        x = F.gelu(x)
        #x = torch.tanh(x)

        x2 = self.w1(x)
        x =  x2
        x =F.gelu(x)
        #x = torch.tanh(x)


        x2 = self.w2(x)
        x =  x2
        x = F.gelu(x)
        #x = torch.tanh(x)

        x2 = self.w3(x)
        x =  x2 
        x = F.gelu(x)
        #x = torch.tanh(x)

        x = x.permute(0, 2, 3, 1)
        
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
    
    
class CONV(nn.Module):
    def __init__(self):
        super(CONV, self).__init__()
        
        self.fc0 = nn.Linear(2 ,64)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 ,1)


                
        


    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.pool(torch.tanh(self.conv1(x)))
        
        x = self.pool(torch.tanh(self.conv2(x)))
       
        x = self.pool(torch.tanh(self.conv3(x)))    
        
        x = self.pool(torch.tanh(self.conv4(x)))
        
        x = torch.tanh(self.conv5(x))
        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.fc1(x)
        return x