

def callPDE(PDE_name):
    if PDE_name == 'PDE1':
        from .PDE1 import operator, f, u_exact, load_data
        return operator, f, u_exact, load_data
    elif PDE_name == 'PDE2':
        from .PDE2 import operator, f, u_exact, load_data
        return operator, f, u_exact, load_data
    elif PDE_name == 'PDE3':
        from .PDE3 import operator, f, u_exact, load_data
        return operator, f, u_exact, load_data
    
def callData(Dataloader_name):
    if Dataloader_name == 'Dataloader':
        from .Dataloader import Train_Test_loaders
        return Train_Test_loaders