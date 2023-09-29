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
args = parser.parse_args()
random_seed = int(args.seed)
set_seed(random_seed)
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

#########################################################################

################## Creating the Models ##################################

models = {
'FNO': FNO2d(modes1=12, modes2=12, width=32),
'CNN': ResNet50(out_shape=(28,28,1), channels=3),
'WNO': WNO2d(shape=(28,28), level=3, width=128),
'UNET': UNet(enc_chs=(64, 128, 256, 512), dec_chs=(512, 256, 128, 64), output_shape=(28,28)),
'FNN': FNN(in_shape=(28,28,3), hidden_features=256, hidden_layers=1, out_shape=(28,28,1)),
}

#########################################################################

###################### Dataset Params ###################################


PATH = 'data/Compression_N70000_D28.npz'
training_data_resolution = 28
grid_size = 28
ntrain = 50000
nval = 10000
ntest = 10000  
batch_size = 20

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

out_transform = transforms.OutTransforms(y_normalizer, device=device).stdTransform


for model_name in models:
    model = models[model_name].to(device)
    if model_name=='WNO':
        hyperparameters['gamma'] = 0.75
        hyperparameters['step_size'] = 50
    trainer = Trainer(model_name=f"Benchmark+{model_name}+Compression", model=model, hyperparams=hyperparameters, \
                    output_transform=out_transform, device=device)
    trainer.fit(train_dataloader=train_grid_loader, val_dataloader=val_grid_loader, test_dataloader=test_grid_loader)

#########################################################################

