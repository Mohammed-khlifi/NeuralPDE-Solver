import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def u_pred(x, y, model):
    return model(torch.cat((x, y), dim=-1))

def u_exact(x, y):
    return torch.exp(-10*(x**2 + y**2))

def calculate_laplacian(x, y, model):
    x.requires_grad_(True)
    y.requires_grad_(True)

    u = u_pred(x, y, model)

    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

    return u_xx + u_yy

def function(x, y): 
    return torch.exp(-10 * (x**2 + y**2))*(-20)*(1-20*x**2) + torch.exp(-10 * (x**2 + y**2))*(-20)*(1-20*y**2)


def risidual(x, y, model):

    f_pred = calculate_laplacian(x, y, model)
    f_true = function(x, y)
    risidual = (f_pred - f_true) 

    return torch.abs(risidual)


def adaptive_collocation_points(x, y, model, k=1.0):

    risiduals = risidual(x, y, model)

    p = risiduals**k / torch.sum(risiduals**k)
    indices = torch.multinomial(p.flatten(), num_samples=p.numel(), replacement=True)

    x_new = x.flatten()[indices].unsqueeze(1)
    y_new = y.flatten()[indices].unsqueeze(1)

   
    return x_new, y_new

def create_grid(num_points):
    x = np.linspace(-3, 3, num_points)
    y = np.linspace(-3, 3, num_points)
    x_grid, y_grid = np.meshgrid(x, y)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    return x_grid, y_grid

#Randomly select points from the grid
def select_random_points(x_grid, y_grid, num_samples):
    indices = np.random.choice(len(x_grid), num_samples, replace=False)
    selected_x = x_grid[indices]
    selected_y = y_grid[indices]
    return selected_x, selected_y


def loss_boundary_left(x, model):
    
    u_pred_boundary = u_pred(x, -3*torch.ones_like(x), model)
    u_boundary =  torch.exp(-10*(x**2+9)).view_as(u_pred_boundary)

    return torch.mean((u_pred_boundary- u_boundary)**2)


def loss_boundary_up(x, model):
    
    u_pred_boundary = u_pred(x, 3*torch.ones_like(x), model)
    u_boundary =  torch.exp(-10*(x**2+9)).view_as(u_pred_boundary)

    return torch.mean((u_pred_boundary - u_boundary)**2)


def loss_boundary_right(y, model):

    u_pred_boundary = u_pred(-3*torch.ones_like(y), y, model)
    u_boundary =  torch.exp(-10*(y**2+9)).view_as(u_pred_boundary)

    return torch.mean((u_pred_boundary - u_boundary)**2)


def loss_boundary_down(y, model):
    
    u_pred_boundary = u_pred(3*torch.ones_like(y), y, model)
    u_boundary =  torch.exp(-10*(y**2+9)).view_as(u_pred_boundary)

    return torch.mean((u_pred_boundary - u_boundary)**2)


def loss_of_pde(x, y, model):

    f_pred = calculate_laplacian(x, y, model)
    f_true = function(x, y).view_as(f_pred)

    return torch.mean((f_pred - f_true )**2)


def loss_function(x, y, model ,w1=1.0, w2=1.0, w3=1.0):

    loss_b_l = loss_boundary_left(x, model)
    loss_b_u = loss_boundary_up(x, model)
    loss_b_r = loss_boundary_right(y, model)
    loss_b_d = loss_boundary_down(y, model)
    loss_pde = loss_of_pde(x, y, model)

    return loss_b_l + loss_b_u + loss_b_r + loss_b_d + loss_pde , w1, w2, w3

def loss_function_with_adaptive_collocation_points(x, y,x_adaptive,y_adaptive, model,w1 = 1,w2 =1,w3=1, k=1.0):

    loss_b_l = loss_boundary_left(x, model)
    loss_b_u = loss_boundary_up(x, model)
    loss_b_r = loss_boundary_right(y, model)
    loss_b_d = loss_boundary_down(y, model)
    loss_pde = loss_of_pde(x, y, model)

    loss_adaptive = loss_of_pde(x_adaptive, y_adaptive, model)

    return loss_b_l + loss_b_u + loss_b_r + loss_b_d + loss_pde + loss_adaptive , w1, w2, w3

def loss_function_with_weights(x, y, model, w1, w2, w3):

    loss_b_l = loss_boundary_left(x, model)
    loss_b_u = loss_boundary_up(x, model)
    loss_b_r = loss_boundary_right(y, model)
    loss_b_d = loss_boundary_down(y, model)
    loss_pde = loss_of_pde(x, y, model)

    return (torch.exp(-w1)*(loss_b_l + loss_b_d) + torch.exp(-w2)*(loss_b_u + loss_b_r) + torch.exp(-w3)*loss_pde +w1**2 + w2**2 + w3**2)*0.5 , w1, w2, w3

def loss_function_with_weights_and_adaptive_collocation_points(x, y,x_adaptive,y_adaptive, model, w1, w2, w3, k=1.0):

    loss_b_l = loss_boundary_left(x, model)
    loss_b_u = loss_boundary_up(x, model)
    loss_b_r = loss_boundary_right(y, model)
    loss_b_d = loss_boundary_down(y, model)
    loss_pde = loss_of_pde(x, y, model)

    loss_adaptive = loss_of_pde(x_adaptive, y_adaptive, model)

    return (torch.exp(w1)*(loss_b_l + loss_b_d) + torch.exp(w2)*(loss_b_u + loss_b_r) + torch.exp(w3)*(loss_pde+ loss_adaptive) + w1**2 + w2**2 + w3**2 )*0.5 , w1, w2, w3

def train_pinn(x, y , model , optimizer,schedular ,epochs,loss_function ,  w1=torch.tensor(1.0, requires_grad=True), w2=torch.tensor(1.0, requires_grad=True), w3=torch.tensor(1.0, requires_grad=True)):
    
    MSE =[]
    LOSS = []
    L2_error = []

    W1 = []
    W2 = []
    W3 = []

    for epoch in range(epochs):

        optimizer.zero_grad()
        loss,w1,w2,w3 = loss_function(x, y, model, w1, w2, w3)
        W1.append(torch.exp(-w1).detach().numpy())
        W2.append(torch.exp(-w2).detach().numpy())
        W3.append(torch.exp(-w3).detach().numpy())
        loss.backward()
        optimizer.step()
        schedular.step()

        if epoch % 100 == 0:
            
            
            u_pred_values = u_pred(x, y, model)
            u_exact_values = u_exact(x, y)
            
            mse = torch.mean((u_pred_values - u_exact_values)**2)
            l2_error = torch.linalg.norm(u_pred_values - u_exact_values) / torch.linalg.norm(u_exact_values)
            L2_error.append(l2_error)
            LOSS.append(loss.item())
            MSE.append(mse.item())
            print(f'Epoch {epoch}, Loss {loss.item()}, mse : {MSE[-1]},l2_error: {L2_error[-1]}')


    return model,LOSS , MSE , L2_error, W1, W2, W3

def train_pinn_with_adaptive_collocation_points(x, y,x_adaptive , y_adaptive, model , optimizer,schedular ,epochs,loss_function,update_rate , w1=torch.tensor(1.0, requires_grad=True), w2=torch.tensor(1.0, requires_grad=True), w3=torch.tensor(1.0, requires_grad=True)):
    
    MSE =[]
    LOSS = []
    L2_error = []
    
    W1 = []
    W2 = []
    W3 = []

    for epoch in range(epochs):

        optimizer.zero_grad()
        
        loss,w1,w2,w3 = loss_function(x, y,x_adaptive,y_adaptive, model, w1, w2, w3)
        W1.append(torch.exp(-w1).detach().numpy())
        W2.append(torch.exp(-w2).detach().numpy())
        W3.append(torch.exp(-w3).detach().numpy())
        loss.backward(retain_graph=True)
        optimizer.step()
        schedular.step()

        if epoch % 100 == 0:

            
            u_pred_values = u_pred(x, y, model)
            u_exact_values = u_exact(x, y)
            mse = torch.mean((u_pred_values - u_exact_values)**2)
            l2_error = torch.linalg.norm(u_pred_values - u_exact_values) / torch.linalg.norm(u_exact_values)
            LOSS.append(loss.item())
            L2_error.append(l2_error.item())
            MSE.append(mse.item())
            print(f'Epoch {epoch}, Loss {loss.item()}, mse : {MSE[-1]},l2_error: {L2_error[-1]}')

        if epoch % update_rate == 0:
            x_adaptive, y_adaptive = adaptive_collocation_points(x_adaptive, y_adaptive, model)
            x_adaptive ,y_adaptive = select_random_points(x_adaptive, y_adaptive, 1000)
            x_adaptive, y_adaptive = torch.tensor(x_adaptive).float(),torch.tensor(y_adaptive).float()
            x_adaptive.requires_grad = True
            y_adaptive.requires_grad = True


    return  model,LOSS , MSE , L2_error ,W1, W2, W3

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


    



    
    
    

    