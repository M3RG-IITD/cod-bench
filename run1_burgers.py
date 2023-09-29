                    # Testing 5 Models:
                    # 1. FNO
                    # 2. WNO
                    # 3. CNN 
                    # 4. UNET 
                    # 5. FNN 

################ Importing Libraries ####################################

from models.fno.fno1d import FNO1d
from models.unet.unet import UNet
from models.wno.wno1d import WNO1d
from models.cnn.resnet import ResNet50
from models.fnn.fnn import FNN

import torch
from trainer import Trainer
from utils import utilities, transforms
from loader.dataloader import *
from loader.loader_1d import *
from utils.utilities import set_seed
import argparse
#########################################################################

############### Setting the Device and Random SEED ######################
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=999, help = "Input Experiment Random Seed")
args = parser.parse_args()
random_seed = int(args.seed)
set_seed(random_seed)
device = torch.device('cuda:6') if torch.cuda.is_available() else 'cpu'

#########################################################################

################## Creating the Models ##################################

models = {
# 'FNO': FNO1d().to(device),
# 'CNN': ResNet50(out_shape=(8192,1), channels=2, dim=1).to(device),
# 'WNO': WNO1d(device=device).to(device),
'UNET': UNet(in_channels=2, output_shape=(8192), dim=1).to(device),
# 'FNN': FNN(in_shape=(8192,2), hidden_features=256, hidden_layers=1, out_shape=(8192,1)).to(device),
}

#########################################################################

###################### Dataset Params ###################################

PATH = 'data/Burgers_N2048_D8192.npz'
training_data_resolution = 8192
grid_size = 8192
batch_size = 20
ntrain = 1700
nval = 148
ntest = 200  

#########################################################################

##################### Generate Data-Loaders #############################

loader = npzloader(path=PATH)

x_train, y_train, x_val, y_val, x_test, y_test = loader.split(ntrain, nval, ntest)

x_normalizer = None#utilities.UnitGaussianNormalizer(x_train)
y_normalizer = None#utilities.UnitGaussianNormalizer(y_train)

# train loader obj
train_obj = DataLoader_1D(X=x_train, y=y_train, n=ntrain, res=training_data_resolution, \
                            grid_size=grid_size, batch_size=batch_size, x_normalizer=x_normalizer)

# val loader obj
val_obj = DataLoader_1D(X=x_val, y=y_val, n=nval, res=training_data_resolution, \
                          grid_size=grid_size, batch_size=batch_size, x_normalizer=x_normalizer)

# test loader obj
test_obj = DataLoader_1D(X=x_test, y=y_test, n=ntest, res=training_data_resolution, \
                           grid_size=grid_size, batch_size=batch_size, x_normalizer=x_normalizer)

# dataloaders with grid info
train_grid_loader = train_obj.get_grid_loader()
val_grid_loader = val_obj.get_grid_loader()
test_grid_loader = test_obj.get_grid_loader()

#########################################################################

################## HyperParameters for Training #########################

hyperparameters = {
    'lr': 1e-3,
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

out_transform = None#transforms.OutTransforms(y_normalizer=y_normalizer, device=device).stdTransform

for model_name in models:
    model = models[model_name]
    if model_name=='UNET':
        hyperparameters['lr'] = 1e-4
    trainer = Trainer(model_name=f"Tuned_Benchmark+{model_name}+Burgers", model=model, hyperparams=hyperparameters, \
                    output_transform=out_transform, device=device)
    trainer.fit(train_dataloader=train_grid_loader, val_dataloader=val_grid_loader, test_dataloader=test_grid_loader)

#########################################################################

