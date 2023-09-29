                    # Testing 5 Models:
                    # 1. GNOT

################ Importing Libraries ####################################

from models.gnot.cgpt import CGPTNO

import torch
from trainer import Trainer
from utils import utilities, transforms
from loader.dataloader import *
from loader.loader_gnot import *
from utils.utilities import set_seed
import argparse
#########################################################################

############### Setting the Device and Random SEED ######################
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, help = "Input Experiment Random Seed")
parser.add_argument("--gpu_card", default=0, help = "GPU CARD")
parser.add_argument("--lr", default=1e-4, help = "OPTIMIZER LEARNING RATE")

args = parser.parse_args()
random_seed = int(args.seed)
gpu = int(args.gpu_card)
lr = float(args.lr)
set_seed(random_seed)
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else 'cpu'

#########################################################################

################## Creating the Models ##################################

models = {
'GNOT': CGPTNO(trunk_size=4, output_size=1, branch_sizes=[]).to(device),
}

#########################################################################

###################### Dataset Params ###################################

PATH = 'data/Stress_N1200_D48.npz'
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
x_normalizer = utilities.UnitGaussianNormalizer(x_train.reshape(ntrain, -1, 1))
y_normalizer = utilities.UnitGaussianNormalizer(y_train.reshape(ntrain, -1))

train_dataset = GNOTDataset(x=x_train, y=y_train, n=ntrain, res=training_data_resolution, \
                            grid_size=grid_size, x_normalizer=x_normalizer)
val_dataset = GNOTDataset(x=x_val, y=y_val, n=nval, res=training_data_resolution, \
                          grid_size=grid_size, x_normalizer=x_normalizer)
test_dataset = GNOTDataset(x=x_test, y=y_test, n=ntest, res=training_data_resolution, \
                           grid_size=grid_size, x_normalizer=x_normalizer)
train_loader = GNOTDataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = GNOTDataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = GNOTDataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
#########################################################################

################## HyperParameters for Training #########################

hyperparameters = {
    'lr': lr,
    'weight_decay': 1e-4,
    'total_steps': 50000,
    'div_factor': 1e4,
    'final_div_factor': 1e4,
    'optimizer': 'AdamW',
    'scheduler': 'OneCycleLR',
    'loss_fn': 'WRelL2',
    'loss_metric': 'MSE',
    'batch_size': batch_size,
    'random_seed': random_seed,
}

#########################################################################

############# Create the Trainer, Fit Dataset and Test ##################

out_transform = transforms.OutTransforms(y_normalizer, device=device).stdTransform

for model_name in models:
    model = models[model_name]
    trainer = Trainer(model_name=f"Benchmark+{model_name}+Stress", model=model, hyperparams=hyperparameters, \
                    output_transform=out_transform, patience=150, device=device)
    trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader)

#########################################################################

