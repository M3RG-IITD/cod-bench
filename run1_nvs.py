                    # Testing 5 Models:
                    # 1. FNO
                    # 2. WNO
                    # 3. CNN 
                    # 4. UNET 
                    # 5. FNN 

################ Importing Libraries ####################################

from models.fno.fno2d import FNO2d
from models.unet.unet import UNet
from models.wno.wno2d import WNO2d
from models.cnn.resnet import ResNet50
from models.fnn.fnn import FNN

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
parser.add_argument("--gpu_card", default=0, help = "GPU CARD")
parser.add_argument("--lr", default=1e-3, help = "OPTIMIZER LEARNING RATE")

args = parser.parse_args()
random_seed = int(args.seed)
gpu = int(args.gpu_card)
lr = float(args.lr)
set_seed(random_seed)
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else 'cpu'

#########################################################################

################## Creating the Models ##################################

models = {
# 'FNO': FNO2d(modes1=16, modes2=16, width=64, in_channels=7).to(device),
# 'CNN': ResNet50(out_shape=(64,64,1), channels=7).to(device),
'WNO': WNO2d(level=3, width=26, in_channels=7, shape=(64,64), device=device).to(device),
# 'UNET': UNet(in_channels=7, output_shape=(64, 64)).to(device),
# 'FNN': FNN(in_shape=(64,64,7), hidden_features=256, hidden_layers=1, out_shape=(64,64,1)).to(device),
}

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
train_grid_loader = train_obj.get_grid_loader()
val_grid_loader = val_obj.get_grid_loader()
test_grid_loader = test_obj.get_grid_loader()

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

out_transform = None#transforms.OutTransforms(y_normalizer, device=device).stdTransform

for model_name in models:
    model = models[model_name]
    if model_name=='WNO':
        hyperparameters['gamma'] = 0.75
        hyperparameters['step_size'] = 50
    if model_name=='UNET':
        hyperparameters['lr'] = 1e-4
    trainer = Trainer(model_name=f"Tuned_Benchmark+{model_name}+NVS", model=model, hyperparams=hyperparameters, \
                    output_transform=out_transform, time_steps=16, device=device)
    trainer.fit_evolution(train_dataloader=train_grid_loader, val_dataloader=val_grid_loader, test_dataloader=test_grid_loader)

#########################################################################

