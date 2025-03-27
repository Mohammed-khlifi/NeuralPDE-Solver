from Models import callmodel
from PDEs import load_PDE, load_dataset
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Neural PDE Solvers')
    parser.add_argument('--model_type', type=str, choices=['PINN', 'NO'],
                      default='PINN', help='Model type (PINN or Neural Operator)')
    parser.add_argument('--model_name', type=str, default='1D_PINNmodel',
                      help='Specific model architecture')
    parser.add_argument('--config', type=str, default='Config/params.yml',
                      help='Path to config file')
    parser.add_argument('--wandb_logs', action='store_true',
                      help='Enable wandb logging')
    parser.add_argument('--PDE', type=str, default='1DPoisson',
                      help='PDE to solve')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--AC', type = int , default = 0,
                      help='Adaptive nodes')
    parser.add_argument('--adaptive_weights', type = int , default = 0,
                      help='Adaptive weights')
    parser.add_argument('--update_rate', type = int , default = 0,
                      help='Update rate')
    parser.add_argument('--Dataset', type=str, default='darcy_flow',
                      help='Dataset to use')
    parser.add_argument('--save_version', type= int, default=False,
                      help='Save model version')
    
    return parser.parse_args()

def train_pinn(model, x):
    mse, loss = model.fit(x)
    return mse, loss

def train_neural_operator(model, train_loader, test_loaders):
    train_losse, val_losse = model.fit(train_loader, test_loaders)
    return train_losse, val_losse

def main():
    args = parse_args()
    with open(args.config, "r") as file:
        param = yaml.safe_load(file)
    
    param.update({
        'save_version': args.save_version,
        'model_name': args.model_name,
        'epochs': args.epochs if args.epochs > 1 else param['epochs'],    
        'lr': args.lr if args.lr > 0 else param['lr'],
        'TF': args.adaptive_weights,
        'wandb_logs': args.wandb_logs,
        'adaptive_nodes': args.AC,
        'update_rate' : args.update_rate
    })

    # Initialize PDE and data
    operator, f, u_exact, load_data = load_PDE(args.PDE)
    
    if args.model_type == 'PINN':
        # PINN specific setup
        x = load_data()['input']
        if isinstance(x, list):
            param['N_input'] = len(x)
        PINNmodel = callmodel(args.model_name)
        if args.epochs == 0:
            param['epochs'] = args
        model = PINNmodel(param=param, operator=operator, f=f, u_exact=u_exact, load_data=load_data)
        mse, loss = train_pinn(model, x)
        print(f"Training completed. MSE: {mse:.4e}, Loss: {loss:.4e}")
        
    elif args.model_type == 'NO':
        # Load NO specific data
        train_loader, test_loaders = load_dataset(args.Dataset)
        model = callmodel(args.model_name)(param=param)
        train_loss, val_loss = train_neural_operator(model, train_loader, test_loaders)
        print(f"Training completed. Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}")
    else :
        raise ValueError("Model type not supported")

if __name__ == "__main__":
    main()
