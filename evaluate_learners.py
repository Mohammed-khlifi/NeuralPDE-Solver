import pandas as pd
from Models.NO_models import FNO_model, CNN_model, UNO_model, TFNO_model
from Models.PINO import PINO_darcy, PINO_poisson
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from PDEs import load_dataset
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Cross validation for NO models')
    parser.add_argument('--Dataset', type=str, choices=['Poisson', 'darcy_flow'],
                      default='Poisson', help='Dataset to use')
    parser.add_argument('--trials', type=int, default=1,
                      help='Number of trials to run')
    parser.add_argument('--config', type=str, default='Config/NO_params.yml', help='Path to config file')
    
    return parser.parse_args()

def evaluate_learners(models , loader , num_trials=1 , param = None):
    param['epochs'] = 1

    kfold = KFold(n_splits=num_trials, shuffle=True)
    results = {model.__name__: [] for model in models}

    AVG_performance = []
    STD_performance = []
    
    train_scores = []
    val_scores = []

    for model_class in models:
        print(f"Training {model_class.__name__}")
        train_fold_scores = []
        val_fold_scores = []
        
        # Perform k-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(loader.dataset)):
            print(f"Fold {fold+1}")
            
            # Create samplers for train/validation split
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            # Create data loaders
            train_loader_k = DataLoader(
                loader.dataset,
                batch_size=32,
                sampler=train_sampler
            )
            val_loader_k = DataLoader(
                loader.dataset,
                batch_size=32,
                sampler=val_sampler
            )
            
            val_loaders_k = {16: val_loader_k}
            
            # Initialize and train model
            Model = model_class(param = param)
            train_score, val_score = Model.fit(train_loader_k, val_loaders_k)
            
            train_fold_scores.append(train_score)
            val_fold_scores.append(val_score)
        
        # Calculate and display average performance
        avg_performance = np.mean(train_scores)
        std_performance = np.std(train_scores)
        print(f"{model_class.__name__} Average Performance: {avg_performance:.4f} ± {std_performance:.4f}")
        
        AVG_performance.append(avg_performance)
        STD_performance.append(std_performance)
        
        train_scores.append(train_fold_scores)
        val_scores.append(val_fold_scores)
        
    results_df = pd.DataFrame(columns=[model.__name__ for model in models], data=[train_scores, val_scores])
    return train_scores , val_scores

def main():
    args = parse_args()
    models = [FNO_model, CNN_model, UNO_model, TFNO_model]
    
    train_loader, test_loaders = load_dataset(args.Dataset)
    if args.Dataset == 'darcy_flow':
        models.append(PINO_darcy)
    elif args.Dataset == 'Poisson':
        models.append(PINO_poisson)
    
    #train
    with open(args.config, "r") as file:
        param = yaml.safe_load(file)
    train_scores , val_scores = evaluate_learners(models, train_loader, num_trials= args.trials, param = param)
    print(train_scores)
    print(val_scores)
    
    
if __name__ == "__main__":
    main()
