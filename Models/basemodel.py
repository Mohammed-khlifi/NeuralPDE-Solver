import torch
import typing as tp
from datetime import datetime
import os
import json
import yaml
from Training.trainer import Trainer


def l2_error(pred, true):
    return torch.sqrt(torch.sum((pred - true) ** 2))/torch.sqrt(torch.sum(true**2))

class Basemodel:


    def __init__(self , param : tp.Dict , operator = None , f = None , u_exact = None , load_data = None) :

        self.base_dir = "trained_models"
        os.makedirs(self.base_dir, exist_ok=True)
        

        self.operator = operator
        self.f = f
        self.u_exact = u_exact
        self.param = param
        self.model = None
        self.optimizer = None
        self.load_data = load_data
        self.pde_configurations = None
        self.boundary_conditions = []

    def fit(self , inputs) :
        if len(inputs) == 2:
            x,y = inputs
            x = torch.sort(x)[0]
            x.requires_grad = True 
            y = torch.sort(y)[0]
            y.requires_grad = True
            x, y = torch.meshgrid(x, y , indexing="ij")

            inputs = [x,y]
        elif len(inputs) == 3:
            x,y,z = inputs
            x = torch.sort(x)[0]
            x.requires_grad = True 
            y = torch.sort(y)[0]
            y.requires_grad = True
            z = torch.sort(z)[0]
            z.requires_grad = True
            x, y, z = torch.meshgrid(x, y, z , indexing="ij")

            inputs = [x,y,z]
        else:
            inputs = torch.sort(inputs)[0]
            inputs.requires_grad = True
            inputs = [inputs]

        
        torch.random.manual_seed(0)
        trainer = Trainer(
            inputs, self.boundary_conditions, self.pde_configurations, model=self.model , watch=self.param['wandb_logs'] , name= self.param['model_name'], lr=self.param['lr'])

        mse, loss = trainer.train(
            epochs=self.param['epochs'], rate=self.pde_configurations.update_rate, loss_function=self.pde_configurations.pde_loss
        )


        self.model = trainer.model

        if self.param['save_version']: 
            #print(self.param['save_version'])
            self.save_version(self.model, {"mse": mse, "loss": loss})

        return mse, loss



    def predict(self , inputs) :
        return self.model(inputs)

    def create_version_dir(self, version):
        """Create directory structure for new model version"""
        version_dir = os.path.join(self.base_dir, f"v{version}")
        os.makedirs(version_dir, exist_ok=True)
        os.makedirs(os.path.join(version_dir, "logs"), exist_ok=True)
        return version_dir
    
    def save_version(self, model, metrics, version=None):
        """Save model version with all artifacts"""
        # Generate version if not provided
        if not version:
            existing_versions = [d for d in os.listdir(self.base_dir) 
                               if os.path.isdir(os.path.join(self.base_dir, d))]
            version = f"1.{len(existing_versions)}"
            
        version_dir = self.create_version_dir(version)
        
        # Save model weights
        torch.save(model.state_dict(), 
                  os.path.join(version_dir, "model.pt"))
        
        # Save config
        with open(os.path.join(version_dir, "config.yml"), "w") as f:
            yaml.dump(self.param, f)
            
        # Save metrics
        results = {
            "metrics": metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.param["model_name"]
        }
        with open(os.path.join(version_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
            
        return version_dir    

    def load_version(self, version):
        """Load model version"""
        version_dir = os.path.join(self.base_dir, f"v{version}")
        if not os.path.exists(version_dir):
            raise ValueError(f"Version v{version} not found")
            
        # Load config
        with open(os.path.join(version_dir, "config.yml"), "r") as f:
            config = yaml.safe_load(f)
            
        # Load model weights
        weights_path = os.path.join(version_dir, "model.pt")
        weights = torch.load(weights_path)
        
        return config, weights
    
    def get_optimizer(self):
        optimizer_type = self.param['optimizer'].lower()
        optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adamw": torch.optim.AdamW,
            "adagrad": torch.optim.Adagrad
        }
        
        if optimizer_type not in optimizers:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        optimizer_class = optimizers[optimizer_type]
        return optimizer_class(self.model.parameters(), lr = self.param['lr'])
    
    
    def get_scheduler(self):
        if self.param['scheduler'] == True:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size= self.param['step_size'], gamma= self.param['gamma'])
            