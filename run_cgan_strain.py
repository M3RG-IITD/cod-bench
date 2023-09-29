                    # Testing 5 Models:
                    # 1. CGAN

################ Importing Libraries ####################################

from models.unet.unet import UNet
from models.cgan.discriminator import Discriminator
import torch
from trainer import Custom_Trainer
from utils import utilities, transforms
from loader.dataloader import *
from loader.loader_2d import *
from utils.utilities import set_seed
import argparse
#########################################################################

############### Setting the Device and Random SEED ######################
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, help = "Input Experiment Random Seed")
parser.add_argument("--gpu", default=0, help = "GPU CARD")
args = parser.parse_args()
random_seed = int(args.seed)
gpu = int(args.gpu)
set_seed(random_seed)
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else 'cpu'

#########################################################################

################## Creating the Models ##################################

Encoder = UNet(output_shape=(48,48)).to(device)
Decoder = Discriminator(channels_img=4).to(device)

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
train_grid_loader = train_obj.get_grid_loader()
val_grid_loader = val_obj.get_grid_loader()
test_grid_loader = test_obj.get_grid_loader()
grid = train_obj.get_grid()

#########################################################################

################## HyperParameters for Training #########################

hyperparameters = {
    'res': 48,
    'enc_lr': 1e-4,
    'dec_lr': 1e-4,
    'dec_weight_decay': 1e-4,
    'enc_weight_decay': 1e-4,
    'enc_optimizer': 'Adam',
    'dec_optimizer': 'Adam',
    'enc_scheduler': 'OneCycleLR',
    'dec_scheduler': 'OneCycleLR',
    'total_steps': 1000,
    'enc_div_factor': 5,
    'dec_final_div_factor': 1e3,
    'dec_div_factor': 5,
    'enc_final_div_factor': 1e3,
    'enc_loss_fn': 'ENC_GAN',
    'dec_loss_fn': 'GAN',
    'loss_metric': 'MSE',
    'batch_size': batch_size,
    'random_seed': random_seed,
}

#########################################################################

############# Create the Trainer, Fit Dataset and Test ##################

out_transform = transforms.OutTransforms(y_normalizer, device=device).stdTransform
in_transform = transforms.InTransforms(grid=grid, device=device)

trainer = Custom_Trainer(model_name='CGAN', project_name='Benchmark+CGAN+Strain', encoder=Encoder, decoder=Decoder, hyperparams=hyperparameters, grid=grid, input_transform=None, \
                         dec_input_transform=in_transform.cganDecoderTransform, output_transform=out_transform, device=device)

trainer.fit(train_dataloader=train_grid_loader, val_dataloader=val_grid_loader, test_dataloader=test_grid_loader)
#########################################################################

