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
    
    else:
        print("model not found ")