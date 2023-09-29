                    # Testing 3 Models:
                    # 1. DEEPONET
                    # 2. PODDEEPONET
                    # 3. SNO 

################ Importing Libraries ####################################

from models.deeponet.deeponet import DeepONet
from models.deeponet.deeponet import PODDeepONet
from models.sno.sno2d import *
from einops import rearrange

import torch
from trainer import Trainer
from utils import utilities, transforms
from loader.dataloader import *
from loader.loader_2d import *
from utils.utilities import set_seed
import argparse
#########################################################################

############### Setting the Device and Random SEED ######################
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, help = "Input Experiment Random Seed")
args = parser.parse_args()
random_seed = int(args.seed)
set_seed(random_seed)
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

#########################################################################

###################### Dataset Params ###################################

PATH = 'data/NavierStokes_V1e-5_N1200_T21.npz'
training_data_resolution = 64
grid_size = 64
batch_size = 20
ntrain = 900
nval = 100 
ntest = 200  

#########################################################################

##################### Generate Data-Loaders #############################

loader = npzloader(path=PATH)

x_train, y_train, x_val, y_val, x_test, y_test = loader.split(ntrain, nval, ntest)

x_normalizer = None#utilities.UnitGaussianNormalizer(x_train)
y_normalizer = None#utilities.UnitGaussianNormalizer(y_train)

# train loader obj
train_obj = DataLoader_2D(X=x_train, y=y_train, n=ntrain, res=training_data_resolution, \
                            grid_size=grid_size, batch_size=batch_size, x_normalizer=x_normalizer)

# val loader obj
val_obj = DataLoader_2D(X=x_val, y=y_val, n=nval, res=training_data_resolution, \
                          grid_size=grid_size, batch_size=batch_size, x_normalizer=x_normalizer)

# test loader obj
test_obj = DataLoader_2D(X=x_test, y=y_test, n=ntest, res=training_data_resolution, \
                           grid_size=grid_size, batch_size=batch_size, x_normalizer=x_normalizer)

# dataloaders with grid info
train_grid_loader = train_obj.get_loader()
val_grid_loader = val_obj.get_loader()
test_grid_loader = test_obj.get_loader()
grid = train_obj.get_grid()

#########################################################################


################## Creating the Models ##################################

models = {
    # 'PODDeepONet': PODDeepONet(BNET_ARCH=(5, (64, 64)), normalized_y_train=rearrange(y_train, 'n h w c -> (n c) h w'), modes=29).to(device),
    # 'DeepONet': DeepONet(BNET_ARCH=(5, (64, 64)), TNET_ARCH=(2,[128,128,64],64),CONST_Y_LOC=grid.to(device),modes=64).to(device),
    'SNO': fSNO2d(sizes=([5, 10, 10], [(64, 201, 201, 64), (33, 101, 101, 33), (10, 10, 10, 10)], [10, 1]), out_shape=[64, 64, 1]).to(device),
}

#########################################################################

################## HyperParameters for Training #########################

hyperparameters = {
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'step_size': 50,
    'gamma': 0.75,
    'optimizer': 'AdamW',
    'scheduler': 'StepLR',
    'loss_fn': 'RelL2',
    'loss_metric': 'MSE',
    'batch_size': batch_size,
    'random_seed': random_seed,
}

#########################################################################

############# Create the Trainer, Fit Dataset and Test ##################

out_transform = None#transforms.OutTransforms(y_normalizer, device=device, modes=32, y_mean=models['PODDeepONet'].__getMean__()).podnetTransform

for model_name in models:
    model = models[model_name]
    # if model_name == 'SNO':
    #     hyperparameters['lr'] = 1e-3
    #     hyperparameters['step_size'] = 50
    #     hyperparameters['gamma'] = 0.75
    trainer = Trainer(model_name=f"Tuned_Benchmark+{model_name}+NVS", model=model, hyperparams=hyperparameters, \
                    output_transform=out_transform, time_steps=16, device=device, is_grid_appended=False)
    trainer.fit_evolution(train_dataloader=train_grid_loader, val_dataloader=val_grid_loader, test_dataloader=test_grid_loader)
    out_transform = None#transforms.OutTransforms(y_normalizer, device=device).stdTransform


#########################################################################

