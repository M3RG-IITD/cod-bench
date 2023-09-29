                    # Testing 3 Models:
                    # 1. DEEPONET
                    # 2. PODDEEPONET
                    # 3. SNO 

################ Importing Libraries ####################################

from models.deeponet.deeponet import DeepONet
from models.deeponet.deeponet import PODDeepONet
from models.sno.sno2d import *

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
parser.add_argument("--gpu", default=5, help = "GPU CARD")
parser.add_argument("--lr", default=1e-4, help = "OPTIMIZER LEARNING RATE")

args = parser.parse_args()
random_seed = int(args.seed)
gpu = int(args.gpu)
lr = float(args.lr)
set_seed(random_seed)
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else 'cpu'

#########################################################################

###################### Dataset Params ###################################

PATH = 'data/Strain_N1200_D48.npz'
training_data_resolution = 48
grid_size = 48
batch_size = 20
ntrain = 900
nval = 100 
ntest = 200  

#########################################################################

##################### Generate Data-Loaders #############################

loader = npzloader(path=PATH)

x_train, y_train, x_val, y_val, x_test, y_test = loader.split(ntrain, nval, ntest)

x_normalizer = utilities.UnitGaussianNormalizer(x_train)
y_normalizer = utilities.UnitGaussianNormalizer(y_train)

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
    # 'DeepONet': DeepONet(BNET_ARCH=(1, (48,48)), TNET_ARCH=(2, [256, 256, 256, 256, 256], 48), \
    #                      CONST_Y_LOC=grid.to(device), modes=256).to(device),
    'PODDeepONet': PODDeepONet(BNET_ARCH=(1, (48,48)), normalized_y_train=y_train, modes=256).to(device),
    # 'SNO': fSNO2d(sizes=([1, 20, 20, 20], [(48, 101, 101, 101, 101, 48), (25, 51, 51, 51, 51, 25), (20, 20, 20, 20, 20, 20)], [20, 20, 1]),
    #             out_shape=[48,48,1]).to(device),
}

#########################################################################

################## HyperParameters for Training #########################

hyperparameters = {
    'lr': lr,
    'weight_decay': 1e-4,
    'step_size': 100,
    'gamma': 0.5,
    'optimizer': 'Adam',
    'scheduler': 'StepLR',
    'loss_fn': 'RelL2',
    'loss_metric': 'MSE',
    'batch_size': batch_size,
    'random_seed': random_seed,
}

#########################################################################

############# Create the Trainer, Fit Dataset and Test ##################

for model_name in models:
    model = models[model_name]
    if model_name == 'SNO':
        hyperparameters['step_size'] = 100
        hyperparameters['gamma'] = 0.75
        out_transform = transforms.OutTransforms(y_normalizer, device=device, modes=256, y_mean=model.__getMean__()).podnetTransform
    else:
        out_transform = transforms.OutTransforms(y_normalizer, device=device).stdTransform
    trainer = Trainer(model_name=f"Inference+{model_name}+Strain", model=model, hyperparams=hyperparameters, \
                    output_transform=out_transform, patience=100, device=device)
    trainer.fit(train_dataloader=train_grid_loader, val_dataloader=val_grid_loader, test_dataloader=test_grid_loader)

#########################################################################

