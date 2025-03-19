import pandas as pd
from Models.NO_models import FNO_model, CNN_model, UNO_model, TFNO_model
from Models.PINO import PINO_darcy, PINO_poisson
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from PDEs import load_PDE, load_dataset
from neuraloperator.neuralop.data.datasets import load_darcy_flow_small
from PDEs.Dataloader import Train_Test_loaders

def evaluate_learners(models , loader):
    param = {
        'epochs': 5,
        'lr': 0.01,
        'save_version': 0
    }

    kfold = KFold(n_splits=5, shuffle=True)
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
    
    """train_loader, test_loaders, _ = load_darcy_flow_small(
            n_train=500, batch_size=32,
            test_resolutions=[16, 32], n_tests=[100, 50],
            test_batch_sizes=[32, 32],
        )"""
        
    train_loaders, test_loaders = Train_Test_loaders(resolution=16, p_min=3, p_max= 100 ,n_samples=556 ,train_size = 0.9, batch_size = 32)
    train_loader = train_loaders[16]
    
    models = [PINO_poisson]
    #models = [PINO_poisson]
    #train
    train_scores , val_scores = evaluate_learners(models, train_loader)
    print(train_scores)
    print(val_scores)
    
    
if __name__ == "__main__":
    main()
# Compare this snippet from Solving-PDE-s-using-neural-network/evaluate_learners.py:
# import pandas as pd
# from src.Models.NO_models import FNO_model, CNN_model, UNO_model, TFNO_model, PINO_darcy
# from sklearn.model_selection import KFold
# from torch.utils.data import DataLoader, SubsetRandomSampler