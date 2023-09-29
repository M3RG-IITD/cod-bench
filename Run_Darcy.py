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
from models.deeponet.deeponet import DeepONet
from models.deeponet.deeponet import PODDeepONet
from models.sno.sno2d import *
from models.oformer.oformer import build_model_2d
from models.cgan.discriminator import Discriminator
from models.gnot.cgpt import CGPTNO

import torch
from trainer import Trainer, Custom_Trainer
from utils import utilities, transforms
from loader.dataloader import *
from loader.loader_2d import *
from loader.loader_gnot import *
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


###################### Dataset Params ###################################

PATH = 'data/Darcy_N2000_D47.npz'
training_data_resolution = 47
grid_size = 47
batch_size = 20
ntrain = 850
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


x_normalizer_gnot = utilities.UnitGaussianNormalizer(x_train.view(ntrain, -1, 1))
y_normalizer_gnot = utilities.UnitGaussianNormalizer(y_train.view(ntrain, -1))

train_dataset = GNOTDataset(x=x_train, y=y_train, n=ntrain, res=training_data_resolution, \
                            grid_size=grid_size, x_normalizer=x_normalizer_gnot)
val_dataset = GNOTDataset(x=x_val, y=y_val, n=nval, res=training_data_resolution, \
                          grid_size=grid_size, x_normalizer=x_normalizer_gnot)
test_dataset = GNOTDataset(x=x_test, y=y_test, n=ntest, res=training_data_resolution, \
                           grid_size=grid_size, x_normalizer=x_normalizer_gnot)
train_loader_gnot = GNOTDataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader_gnot = GNOTDataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader_gnot = GNOTDataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# dataloaders with grid info
train_grid_loader = train_obj.get_grid_loader()
val_grid_loader = val_obj.get_grid_loader()
test_grid_loader = test_obj.get_grid_loader()
train_loader = train_obj.get_loader()
val_loader = val_obj.get_loader()
test_loader = test_obj.get_loader()
grid = train_obj.get_grid()

#########################################################################


#########################################################################

################## Creating the Models ##################################

models1 = {
'FNO': FNO2d(modes1=12, modes2=12, width=32).to(device),
'CNN': ResNet50(out_shape=(47,47,1), channels=3).to(device),
'WNO': WNO2d().to(device),
'UNET': UNet().to(device),
'FNN': FNN(in_shape=(47,47,3), hidden_features=256, hidden_layers=1, out_shape=(47,47,1)).to(device),
}

###########################################################################

models2 = {
    'PODDeepONet': PODDeepONet(normalized_y_train=y_train, modes=115).to(device),
    'DeepONet': DeepONet(CONST_Y_LOC=grid.to(device)).to(device),
    'SNO': fSNO2d(sizes=([1, 10, 10], [(47, 51, 51, 47), (24, 51, 51, 24), (10, 10, 10, 10)], [10, 1])).to(device),
}

#########################################################################

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

############ Create the Trainer, Fit Dataset and Test ##################

out_transform = transforms.OutTransforms(y_normalizer, device=device).stdTransform

for model_name in models1:
    model = models1[model_name]
    if model_name=='WNO':
        hyperparameters['gamma'] = 0.75
        hyperparameters['step_size'] = 50
    if model_name=='UNET':
        hyperparameters['lr'] = 1e-4
    trainer = Trainer(model_name=f"Data_Efficiency_0.50+{model_name}+Darcy", model=model, hyperparams=hyperparameters, \
                    output_transform=out_transform, device=device)
    trainer.fit(train_dataloader=train_grid_loader, val_dataloader=val_grid_loader, test_dataloader=test_grid_loader)

#########################################################################
hyperparameters = {
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'step_size': 25,
    'gamma': 0.5,
    'optimizer': 'AdamW',
    'scheduler': 'StepLR',
    'loss_fn': 'RelL2',
    'loss_metric': 'MSE',
    'batch_size': batch_size,
    'random_seed': random_seed,
}


############# Create the Trainer, Fit Dataset and Test ##################

out_transform = transforms.OutTransforms(y_normalizer, device=device, modes=115, y_mean=models2['PODDeepONet'].__getMean__()).podnetTransform

for model_name in models2:
    model = models2[model_name]
    if model_name == 'SNO':
        hyperparameters['lr'] = 1e-3
        hyperparameters['step_size'] = 50
        hyperparameters['gamma'] = 0.75
    trainer = Trainer(model_name=f"Data_Efficiency_0.50+{model_name}+Darcy", model=model, hyperparams=hyperparameters, \
                    output_transform=out_transform, patience=150, device=device)
    trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader)
    out_transform = transforms.OutTransforms(y_normalizer, device=device).stdTransform


#########################################################################
hyperparameters = {
    'res': 47,
    'enc_lr': 1e-3,
    'dec_lr': 1e-3,
    'dec_weight_decay': 1e-4,
    'enc_weight_decay': 1e-4,
    'enc_optimizer': 'Adam',
    'dec_optimizer': 'Adam',
    'enc_scheduler': 'OneCycleLR',
    'dec_scheduler': 'OneCycleLR',
    'total_steps': 50000,
    'enc_div_factor': 1e2,
    'dec_final_div_factor': 1e5,
    'dec_div_factor': 1e2,
    'enc_final_div_factor': 1e5,
    'enc_loss_fn': 'O2',
    'dec_loss_fn': 'RelL2',
    'loss_metric': 'MSE',
    'batch_size': batch_size,
    'random_seed': random_seed,
}


###################OFormer Model############################################################

encoder, decoder = build_model_2d(res=47)
encoder, decoder = encoder.to(device), decoder.to(device)

input_transform = transforms.InTransforms(grid=grid.view(1, -1, 2), device=device).oformerEncoderTransform
input_dec_transform = transforms.InTransforms(grid=grid.view(1, -1, 2), device=device).oformerDecoderTransform
output_dec_transform = transforms.OutTransforms(device=device, res=47, y_normalizer=y_normalizer).oformerTransform

trainer = Custom_Trainer(model_name='OFormer', project_name='Data_Efficiency_0.50+OFormer+Darcy', res=47, encoder=encoder, decoder=decoder, \
    hyperparams=hyperparameters, grid=grid.view(1, -1, 2), input_transform=input_transform, dec_input_transform=input_dec_transform, dec_output_transform=output_dec_transform, device=device)
trainer.fit(train_dataloader=train_grid_loader, val_dataloader=val_grid_loader, test_dataloader=test_grid_loader)

############################################################################################
hyperparameters = {
    'res': 47,
    'enc_lr': 1e-4,
    'dec_lr': 1e-4,
    'dec_weight_decay': 1e-4,
    'enc_weight_decay': 1e-4,
    'enc_optimizer': 'Adam',
    'dec_optimizer': 'Adam',
    'enc_scheduler': 'OneCycleLR',
    'dec_scheduler': 'OneCycleLR',
    'total_steps': 80000,
    'enc_div_factor': 5,
    'dec_final_div_factor': 5e2,
    'dec_div_factor': 5,
    'enc_final_div_factor': 5e2,
    'enc_loss_fn': 'ENC_GAN',
    'dec_loss_fn': 'GAN',
    'loss_metric': 'MSE',
    'batch_size': batch_size,
    'random_seed': random_seed,
}


Encoder = UNet().to(device)
Decoder = Discriminator(channels_img=4).to(device)

############# Create the Trainer, Fit Dataset and Test ##################

out_transform = transforms.OutTransforms(y_normalizer, device=device).stdTransform
in_transform = transforms.InTransforms(grid=grid, device=device)

trainer = Custom_Trainer(model_name='CGAN', project_name='Data_Efficiency_0.50+CGAN+Darcy', encoder=Encoder, decoder=Decoder, hyperparams=hyperparameters, grid=grid, input_transform=None, \
                         dec_input_transform=in_transform.cganDecoderTransform, output_transform=out_transform, device=device)

trainer.fit(train_dataloader=train_grid_loader, val_dataloader=val_grid_loader, test_dataloader=test_grid_loader)
#########################################################################

hyperparameters = {
    'lr': 1e-4,
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

models = {
'GNOT': CGPTNO(trunk_size=4, output_size=1, branch_sizes=[]).to(device),
}

############# Create the Trainer, Fit Dataset and Test ##################

out_transform = transforms.OutTransforms(y_normalizer, device=device).stdTransform

for model_name in models:
    model = models[model_name]
    trainer = Trainer(model_name=f"Data_Efficiency_0.50+{model_name}+Darcy", model=model, hyperparams=hyperparameters, \
                    output_transform=out_transform, device=device)
    trainer.fit(train_dataloader=train_loader_gnot, val_dataloader=val_loader_gnot, test_dataloader=test_loader_gnot)

#########################################################################

