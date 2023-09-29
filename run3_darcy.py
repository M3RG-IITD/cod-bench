import torch
from trainer import Custom_Trainer
from utils import utilities, transforms
from utils.utilities import set_seed
from models.oformer.oformer import build_model_2d
from loader.dataloader import *
from loader.loader_2d import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, help = "Input Experiment Random Seed")
parser.add_argument("--gpu_card", default=4, help = "GPU CARD")
parser.add_argument("--lr", default=1e-3, help = "OPTIMIZER LEARNING RATE")

args = parser.parse_args()
random_seed = int(args.seed)
gpu = int(args.gpu_card)
lr = float(args.lr)
set_seed(random_seed)
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else 'cpu'


###################### Dataset Params ###################################

PATH = 'data/Darcy_N2000_D47.npz'
training_data_resolution = 47
grid_size = 47
batch_size = 20
ntrain = 1700
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
train_loader = train_obj.get_grid_loader()
val_loader = val_obj.get_grid_loader()
test_loader = test_obj.get_grid_loader()
grid = train_obj.get_grid()
grid = grid.reshape(1, -1, 2)

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

input_transform = transforms.InTransforms(grid=grid, device=device).oformerEncoderTransform
input_dec_transform = transforms.InTransforms(grid=grid, device=device).oformerDecoderTransform
output_dec_transform = transforms.OutTransforms(device=device, res=47, y_normalizer=y_normalizer).oformerTransform

trainer = Custom_Trainer(model_name='OFormer', project_name='Tuned_Benchmark+OFormer+Darcy', res=47, encoder=encoder, decoder=decoder, \
    hyperparams=hyperparameters, grid=grid, input_transform=input_transform, dec_input_transform=input_dec_transform, dec_output_transform=output_dec_transform, device=device)
trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader)

############################################################################################