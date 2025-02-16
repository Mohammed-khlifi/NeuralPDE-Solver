from src.Models import callmodel
from src.PDEs import callPDE
import yaml
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train PINN models')
    parser.add_argument('--model_name', type=str, default='1D_PINNmodel',
                      help='Model type to train')
    parser.add_argument('--config', type=str, default='params.yml',
                      help='Path to config file')
    parser.add_argument('--wandb_logs', type=bool, default=False,
                      help='Log to wandb')
    parser.add_argument('--PDE', type=str, default='PDE1',
                        help='PDE to solve')
    parser.add_argument('--epochs', type=int, default=0,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--save_version', action='store_true',
                        help='Save model version')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load config
    with open(args.config, "r") as file:
        param = yaml.safe_load(file)
    
    # Initialize PDE
    operator, f, u_exact, load_data = callPDE(args.PDE)
        
    # Initialize model
    PINNmodel = callmodel(args.model_name)
    if args.epochs > 0:
        param['epochs'] = args.epochs
    if args.lr > 0:
        param['lr'] = args.lr
    
    x = load_data()['input']
    if x is list:
        param['N_input'] = len(x)
    param['model_name'] = args.model_name
    model = PINNmodel(param=param, operator=operator, f=f, u_exact=u_exact , load_data=load_data)
    
    # Train
    mse, loss = model.fit(x)
    print(f"Training completed. MSE: {mse:.4e}, Loss: {loss:.4e}")

if __name__ == "__main__":
    main()