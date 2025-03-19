from .models import PINN_Net, CustomPINN

def callmodel(model_name):
    if model_name == '1D_PINNmodel':
        from .PINNModels import PINNModel
        return PINNModel
    elif model_name == '2D_PINNmodel':
        from .PINNModels import PINNModel_2D
        return PINNModel_2D
    elif model_name == '3D_PINNmodel':
        from .PINNModels import PINNModel_3D
        return PINNModel_3D
    elif model_name == 'NO_basemodel':
        from .NO_basemodel import NO_basemodel
        return NO_basemodel
    elif model_name == 'FNO':
        from .FNO_model import FNO_model
        return FNO_model
    elif model_name == 'CNN':
        from .CNN_model import CNN_model
        return CNN_model
    elif model_name == 'UNO':
        from .NO_models import UNO_model
        return UNO_model
    elif model_name == 'TFNO':
        from .NO_models import TFNO_model
        return TFNO_model
    
    elif model_name == 'CODANO':
        from .NO_models import CODANO_model
        return CODANO_model
    
    elif model_name == 'PINO_poisson':
        from .PINO import PINO_poisson
        return PINO_poisson
    
    elif model_name == 'PINO_darcy':
        from .PINO import PINO_darcy
        return PINO_darcy
    
    else:
        print("model not found ")