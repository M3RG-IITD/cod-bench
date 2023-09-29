import torch
import torch.nn as nn
import numpy as np
import wandb
import os
import operator
from functools import reduce
from timeit import default_timer
from utils.optimizers import Adam, AdamW
from utils.utilities import LpLoss, FormerLoss, GANLoss, EncoderLoss, WeightedLpRelLoss, FormerLoss1D
from torch.optim.lr_scheduler import StepLR, OneCycleLR

OPTIMIZERS = {
    "Adam": Adam,
    "AdamW": AdamW,
}

SCHEDULERS = {
    "StepLR": StepLR,
    "OneCycleLR": OneCycleLR,
}

LOSS = {
    "WRelL2": WeightedLpRelLoss(p=1, component="all", normalizer=None),
    "RelL2": LpLoss(), # L2 Loss by default, reduce=mean
    "MSE": nn.MSELoss(), # reduce=mean
    "O2": FormerLoss,
    "O1": FormerLoss1D,
    "GAN": GANLoss(),
    "ENC_GAN": EncoderLoss(),
}

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

class Trainer(object):
    def __init__(self, model_name, model, hyperparams, input_transform=None, output_transform=None, \
                 patience=50, batch_size=20, time_steps=15, is_grid_appended=True, grid_dim=2, device=torch.device('cuda'), epsilon=1e-6, use_same_log_loss=False):
        self.model = model
        self.eps = epsilon
        self.T = time_steps
        self.grid = is_grid_appended
        self.grid_dim = grid_dim
        self.hyperparams = hyperparams
        self.train = self.training_step
        self.validate = self.validation_step
        self.test = self.testing
        self.test_evolution = self.testing_evl
        self.batch_size = batch_size
        self.log_loss_func = LOSS['RelL2']
        self.optimizer = OPTIMIZERS[hyperparams['optimizer']](params=model.parameters(), \
                         lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])
        self.loss_func  = LOSS[hyperparams['loss_fn']]
        if use_same_log_loss:
            self.log_loss_func= self.loss_func
        if 'loss_metric' in hyperparams:
            self.metric_loss_fn = LOSS[hyperparams['loss_metric']]
            self.test_metric_name = hyperparams['loss_metric']
        else:
            self.metric_loss_fn = None
            self.test_metric_name = None
        if hyperparams['scheduler']=='StepLR':
            self.scheduler = SCHEDULERS[hyperparams['scheduler']](optimizer=self.optimizer, \
                                step_size=hyperparams['step_size'], gamma=hyperparams['gamma'])
        else:
            self.scheduler = SCHEDULERS[hyperparams['scheduler']](optimizer=self.optimizer, \
                            max_lr=hyperparams['lr'], total_steps=hyperparams['total_steps'], pct_start=0.2, \
                            div_factor=hyperparams['div_factor'], final_div_factor=hyperparams['final_div_factor'])
        
        self.device = device
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.patience = patience
        self.project_name = model_name + str(hyperparams['lr']) + str(hyperparams['batch_size'])
        self.loss_func_name = hyperparams['loss_fn']
        if 'GNOT' in model_name:
            self.train = self.train_gnot
            self.validate = self.val_gnot
            self.test = self.test_gnot
            self.test_evolution = self.test_evl_gnot
        wandb.init(project=self.project_name)
        num_params = count_params(model)
        ###################Terminal Output############################################################
        print(f'Number of Parameters in the Model: {num_params}')
        print(f"optimizer: {hyperparams['optimizer']}, scheduler: {hyperparams['scheduler']}")
        print(f"loss_fn: {hyperparams['loss_fn']}, test_loss_metric: {hyperparams['loss_metric']}")
        print(f"Initial_LR: {hyperparams['lr']}, weight_decay: {hyperparams['weight_decay']}")
        ################WandB Logging#################################################################
        wandb.log({"Model_Architecture": model, "Num_Params": num_params, "Random_SEED": hyperparams['random_seed'],
        "optimizer": hyperparams['optimizer'], "scheduler": hyperparams['scheduler'], 
        "loss_fn": hyperparams['loss_fn'], "test_loss_metric": hyperparams['loss_metric'],
        "Initial_LR": hyperparams['lr'], "weight_decay": hyperparams['weight_decay']})
        ##############################################################################################

    def fit(self, train_dataloader, val_dataloader, test_dataloader):
        best_val_loss = 10000.0
        test_loss1 = 0.0
        test_loss2 = 0.0
        std1 = 0.0
        std2 = 0.0
        epsilon = self.eps
        learning = self.patience
        epoch = 0
        train_start_timer = default_timer()
        while learning: 
            epoch_start_timer = default_timer()
            learning -= 1
            epoch += 1
            loss = 0
            validation_loss = 0
            for batch in train_dataloader:
                if 'GNOT' in self.project_name:
                    batch.append(batch[0].ndata['y'].squeeze())
                loss_batch, pred = self.train(batch) 
                loss += loss_batch
                if self.hyperparams['scheduler']=='OneCycleLR':  
                    self.scheduler.step()
            if self.hyperparams['scheduler']=='StepLR':       
                self.scheduler.step()
            with torch.no_grad():
                for batch in val_dataloader:
                    if 'GNOT' in self.project_name:
                        batch.append(batch[0].ndata['y'].squeeze())
                    loss_batch, pred = self.validate(batch)  
                    validation_loss += loss_batch   
            loss /= len(train_dataloader)
            validation_loss /= len(val_dataloader)
            #########HACK############################
            if validation_loss < best_val_loss:
                if best_val_loss - validation_loss > epsilon:
                    learning = self.patience
                best_val_loss = validation_loss 
                path = f'./models_state_dict/{self.project_name}'
                os.makedirs(path, exist_ok = True) 
                path += '/model.pt'
                torch.save(self.model.state_dict(), path)        
            #########################################
            epoch_time = np.round((default_timer() - epoch_start_timer), 4)
            wandb.log({"Epoch": epoch, "Time": epoch_time,"Train Loss": loss, "Validation Loss": validation_loss})
            print('Epoch := %s || Time (sec):= %s  || Train Loss := %.3e || Validation Loss := %.3e'\
                  %(epoch, epoch_time, loss, validation_loss))
            
        train_time = np.round((default_timer() - train_start_timer), 4)
        print("\n" + "##################################################")
        print(f"Total Train Time (sec): {train_time}")
        wandb.log({"Total_epochs": epoch})
        print("##################################################")
        wandb.finish()

    def fit_evolution(self, train_dataloader, val_dataloader, test_dataloader):
        best_val_loss = 10000.0
        test_loss1 = 0.0
        test_loss2 = 0.0
        std1 = 0.0
        std2 = 0.0
        epsilon = self.eps
        learning = self.patience
        epoch = 0
        train_start_timer = default_timer()
        while learning:
            epoch_start_timer = default_timer()
            learning -= 1
            epoch += 1
            validation_loss = 0
            for batch in train_dataloader:
                loss = 0
                for i, data in enumerate(batch):
                    batch[i] = data.to(self.device)
                if 'GNOT' in self.project_name:
                    batch.append(batch[0].ndata['y'])
                    data = (batch[0], batch[1], batch[2], batch[3][..., 0])
                else:
                    data = (batch[0], batch[1][..., 0])
                pred_f = torch.zeros(batch[-1].shape).to(self.device)
                for t in range(self.T):
                    loss_batch, pred = self.validate(data) 
                    pred_f[..., t] = pred.squeeze(-1)
                    loss += loss_batch  
                    if t == self.T - 1:
                        break
                    if 'GNOT' in self.project_name:
                        x_t = torch.cat((data[0].ndata['x'][..., 1:-self.grid_dim], pred, data[0].ndata['x'][..., -self.grid_dim:]), dim=-1)
                        data[0].ndata['x'] = x_t
                        data[2].ndata['x'] = x_t
                        data = (data[0], data[1], data[2], batch[3][..., t+1])                    
                    else:
                        if self.grid:
                            data = (torch.cat((data[0][..., 1:-self.grid_dim], pred, data[0][..., -self.grid_dim:]), dim=-1), batch[1][..., t+1])   
                        else:                   
                            data = (torch.cat((data[0][..., 1:], pred), dim=-1), batch[1][..., t+1])  
                    del pred
                    del loss_batch
                    torch.cuda.empty_cache() 
                train_loss = loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                del batch
                del data
                del loss
                torch.cuda.empty_cache()
                if self.hyperparams['scheduler']=='OneCycleLR':  
                    self.scheduler.step()
            if self.hyperparams['scheduler']=='StepLR':       
                self.scheduler.step()
            with torch.no_grad():
                for batch in val_dataloader:
                    for i, data in enumerate(batch):
                        batch[i] = data.to(self.device)
                    if 'GNOT' in self.project_name:
                        batch.append(batch[0].ndata['y'])
                        data = (batch[0], batch[1], batch[2], batch[3][..., 0])
                    else:
                        data = (batch[0], batch[1][..., 0])
                    pred_f = torch.zeros(batch[-1].shape).to(self.device)
                    validation_loss_step = 0
                    for t in range(self.T):
                        loss_batch, pred = self.validate(data)
                        pred_f[..., t] = pred.squeeze(-1) 
                        validation_loss_step += loss_batch.item()
                        if t == self.T - 1:
                            break 
                        if 'GNOT' in self.project_name:
                            x_t = torch.cat((data[0].ndata['x'][..., 1:-self.grid_dim], pred, data[0].ndata['x'][..., -self.grid_dim:]), dim=-1)
                            data[0].ndata['x'] = x_t
                            data[2].ndata['x'] = x_t
                            data = (data[0], data[1], data[2], batch[3][..., t+1])                    
                        else:
                            if self.grid:
                                data = (torch.cat((data[0][..., 1:-self.grid_dim], pred, data[0][..., -self.grid_dim:]), dim=-1), batch[1][..., t+1])   
                            else:                   
                                data = (torch.cat((data[0][..., 1:], pred), dim=-1), batch[1][..., t+1]) 
                    validation_loss += self.log_loss_func(pred_f.view(pred_f.shape[0], -1), batch[-1].view(batch[-1].shape[0], -1))   
                    del batch
                    del data
                    torch.cuda.empty_cache()
            train_loss /= len(train_dataloader)
            validation_loss /= len(val_dataloader)
            #########HACK############################
            if validation_loss < best_val_loss:
                if best_val_loss - validation_loss > epsilon:
                    learning = self.patience
                best_val_loss = validation_loss                    
            #########################################
            epoch_time = np.round((default_timer() - epoch_start_timer), 4)
            wandb.log({"Epoch": epoch, "Time": epoch_time,"Train Loss": train_loss, "Validation Loss": validation_loss})
            print('Epoch := %s || Time (sec):= %s  || Train Loss := %.3e || Validation Loss := %.3e'\
                  %(epoch, epoch_time, train_loss, validation_loss))
            
        train_time = np.round((default_timer() - train_start_timer), 4)
        print("\n" + "##################################################")
        print(f"Total Train Time (sec): {train_time}")
        wandb.log({"Total_epochs": epoch})
        print("##################################################")
        wandb.finish()
        path = f'./models_state_dict/{self.project_name}'
        os.makedirs(path, exist_ok = True) 
        path += '/model.pt'
        torch.save(self.model.state_dict(), path)


    def training_step(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        self.model.train()
        batch_size = x.shape[0]
        if self.input_transform is not None:
            x = self.input_transform(x)
        pred = self.model(x)
        if self.output_transform is not None:
            pred = self.output_transform(pred)
        self.optimizer.zero_grad()
        loss = self.loss_func(pred.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()
        self.optimizer.step()
        return loss, pred
    
    def validation_step(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        self.model.eval()
        batch_size = x.shape[0]
        if self.input_transform is not None:
            x = self.input_transform(x)
        pred = self.model(x)
        if self.output_transform is not None:
            pred = self.output_transform(pred)
        loss = self.log_loss_func(pred.view(batch_size, -1), y.view(batch_size, -1))
        return loss, pred

    def testing(self, test_dataloader):
        start_timer = default_timer()
        loss = 0
        loss2 = 0
        loss_arr = []
        loss2_arr = []
        with torch.no_grad():
            for x, y in test_dataloader:
                batch_size = x.shape[0]
                x, y = x.to(self.device), y.to(self.device)
                if self.input_transform is not None:
                    x = self.input_transform(x)
                pred = self.model(x)
                if self.output_transform is not None:
                    pred = self.output_transform(pred)
                loss_batch = self.log_loss_func(pred.view(batch_size, -1), y.view(batch_size, -1)).item()
                loss += loss_batch
                loss_arr.append(loss_batch)
                if self.metric_loss_fn is not None:
                    loss_batch = self.metric_loss_fn(pred.view(batch_size, -1), y.view(batch_size, -1)).item()
                    loss2 += loss_batch
                    loss2_arr.append(loss_batch)
        loss = loss/len(test_dataloader)
        loss2 = loss2/len(test_dataloader)
        end_timer = default_timer()
        return loss, loss2
    
    def testing_evl(self, test_dataloader):
        loss = 0
        loss2 = 0
        loss_arr = []
        loss2_arr = []
        with torch.no_grad():
            for batch in test_dataloader:
                batch[0] = batch[0].to(self.device)
                batch[1] = batch[1].to(self.device)
                x, y = batch[0], batch[1][..., 0]
                batch_size = x.shape[0]
                pred_f = torch.zeros(batch[1].shape).to(self.device)
                for t in range(self.T):
                    x, y = x.to(self.device), y.to(self.device)
                    if self.input_transform is not None:
                        x = self.input_transform(x)
                    pred = self.model(x)
                    if self.output_transform is not None:
                        pred = self.output_transform(pred)
                    pred_f[..., t] = pred.squeeze(-1)

                    if t == self.T - 1:
                        break
                    if self.grid:
                        x, y = torch.cat((x[..., 1:-self.grid_dim], pred, x[..., -self.grid_dim:]), dim=-1), batch[1][..., t+1] 
                    else:                   
                        x, y = torch.cat((x[..., 1:], pred), dim=-1), batch[1][..., t+1] 
                loss_batch = self.log_loss_func(pred_f.view(batch_size, -1), batch[1].view(batch_size, -1))
                loss += loss_batch
                loss_arr.append(loss_batch)
                if self.metric_loss_fn is not None:
                    loss_batch = self.metric_loss_fn(pred_f.view(batch_size, -1), batch[1].view(batch_size, -1))
                    loss2 += loss_batch
                    loss2_arr.append(loss_batch)
                                                
        loss = loss/len(test_dataloader)
        loss2 = loss2/len(test_dataloader)
        return loss, loss2
    
    def train_gnot(self, data, grad_clip=0.999):
        self.model.train()
        g, u_p, g_u, y = data
        g, g_u, u_p, y = g.to(self.device), g_u.to(self.device), u_p.to(self.device), y.to(self.device)
        if self.input_transform is not None:
            g, u_p, g_u = self.input_transform(g, u_p, g_u)
        pred = self.model(g, u_p, g_u)
        if self.output_transform is not None:
            pred = self.output_transform(pred.reshape(self.batch_size, -1)).reshape(-1)
        loss = self.loss_func(g, pred, y.squeeze())
        # for burgers, weighted loss calculation needs to be optimized, cuda out of memory error 
        # loss = self.log_loss_func(pred.view(self.batch_size, -1), y.view(self.batch_size, -1))
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()
        return loss, pred
    
    def val_gnot(self, data):
        g, u_p, g_u, y = data
        g, g_u, u_p, y = g.to(self.device), g_u.to(self.device), u_p.to(self.device), y.to(self.device)
        if self.input_transform is not None:
            g, u_p, g_u = self.input_transform(g, u_p, g_u)
        pred = self.model(g, u_p, g_u)
        if self.output_transform is not None:
            pred = self.output_transform(pred.reshape(self.batch_size, -1)).reshape(-1)
        loss = self.log_loss_func(pred.view(self.batch_size, -1), y.view(self.batch_size, -1))
        return loss, pred
    
    def test_gnot(self, test_dataloader):
        start_timer = default_timer()
        loss = 0
        loss2 = 0
        loss_arr = []
        loss2_arr = []
        with torch.no_grad():
            for data in test_dataloader:
                g, u_p, g_u = data
                g, g_u, u_p = g.to(self.device), g_u.to(self.device), u_p.to(self.device)
                if self.input_transform is not None:
                    g, u_p, g_u = self.input_transform(g, u_p, g_u)
                y = g.ndata['y'].squeeze()
                pred = self.model(g, u_p, g_u)
                if self.output_transform is not None:
                    pred = self.output_transform(pred.reshape(self.batch_size, -1)).reshape(-1)
                loss_batch = self.log_loss_func(pred.view(self.batch_size, -1), y.view(self.batch_size, -1))
                loss += loss_batch
                loss_arr.append(loss_batch)
                if self.metric_loss_fn is not None:
                    loss_batch = self.metric_loss_fn(pred.view(self.batch_size, -1), y.view(self.batch_size, -1))
                    loss2 += loss_batch
                    loss2_arr.append(loss_batch)
        loss = loss/len(test_dataloader)
        loss2 = loss2/len(test_dataloader)
        end_timer = default_timer()
        return loss, loss2
    
    def test_evl_gnot(self, test_loader):
        loss = 0
        loss2 = 0
        loss_arr = []
        loss2_arr = []
        with torch.no_grad():
            for batch in test_loader:
                    batch.append(batch[0].ndata['y'])
                    for i, data in enumerate(batch):
                        batch[i] = data.to(self.device)
                    g, u_p, g_u, y = batch[0], batch[1], batch[2], batch[3][..., 0]
                    pred_f = torch.zeros(batch[3].shape).to(self.device)
                    for t in range(self.T):
                        g, g_u, u_p, y = g.to(self.device), g_u.to(self.device), u_p.to(self.device), y.to(self.device)
                        if self.input_transform is not None:
                            g, u_p, g_u = self.input_transform(g, u_p, g_u)
                        pred = self.model(g, u_p, g_u)
                        pred_f[..., t] = pred.squeeze(-1)
                        if self.output_transform is not None:
                            pred = self.output_transform(pred.reshape(self.batch_size, -1)).reshape(-1)
                        if t == self.T - 1:
                            break
                        x_t = torch.cat((g.ndata['x'][..., 1:-self.grid_dim], pred, g.ndata['x'][..., -self.grid_dim:]), dim=-1)
                        g.ndata['x'] = x_t
                        g_u.ndata['x'] = x_t
                        y = batch[3][..., t+1] 
                    loss_batch = self.log_loss_func(pred_f.view(pred_f.shape[0], -1), batch[3].view(batch[3].shape[0], -1))
                    loss += loss_batch
                    loss_arr.append(loss_batch)
                    if self.metric_loss_fn is not None:
                        loss_batch = self.metric_loss_fn(pred_f.view(pred_f.shape[0], -1), batch[3].view(batch[3].shape[0], -1))
                        loss2 += loss_batch
                        loss2_arr.append(loss_batch)
                                         
        loss = loss/len(test_loader)
        loss2 = loss2/len(test_loader)
        return loss, loss2

class Custom_Trainer():
    def __init__(self, model_name, project_name, encoder, decoder, hyperparams, grid=None, res=None, input_transform=None, output_transform=None, \
                 dec_input_transform=None, dec_output_transform=None, grid_dim=2, time_steps=15, patience=50, device=torch.device('cuda'), epsilon=1e-6):
        if model_name=='OFormer':
             self.train = self.train_oformer
             self.validate = self.val_oformer
             self.test = self.test_oformer
             self.test_evolution = self.test_evl_oformer

        elif model_name=='CGAN':
             self.train = self.train_cgan
             self.validate = self.val_cgan
             self.test = self.test_cgan
             self.test_evolution = self.test_evl_cgan

        self.T = time_steps
        self.res = res
        self.encoder = encoder
        self.decoder = decoder
        self.grid = grid
        self.grid_dim = grid_dim
        self.eps = epsilon
        self.enc_optimizer = OPTIMIZERS[hyperparams['enc_optimizer']](params=encoder.parameters(), \
                         lr=hyperparams['enc_lr'], weight_decay=hyperparams['enc_weight_decay'])
        self.enc_loss_func  = LOSS[hyperparams['enc_loss_fn']]
        if 'loss_metric' in hyperparams:
            self.metric_loss_fn = LOSS[hyperparams['loss_metric']]
            self.test_metric_name = hyperparams['loss_metric']
        else:
            self.metric_loss_fn = None
            self.test_metric_name = None
        self.enc_scheduler = SCHEDULERS[hyperparams['enc_scheduler']](optimizer=self.enc_optimizer, \
                            max_lr=hyperparams['enc_lr'], total_steps=hyperparams['total_steps'], pct_start=0.2, \
                            div_factor=hyperparams['enc_div_factor'], final_div_factor=hyperparams['enc_final_div_factor'])
        self.device = device
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.patience = patience
        self.project_name = project_name + str(hyperparams['enc_lr']) + str(hyperparams['batch_size'])
        self.loss_func_name = hyperparams['enc_loss_fn']
        wandb.init(project=self.project_name)
        num_params = count_params(encoder)
        ###################Terminal Output############################################################
        print(f'Number of Parameters in the encoder: {num_params}')
        print(f"optimizer: {hyperparams['enc_optimizer']}, scheduler: {hyperparams['enc_scheduler']}")
        print(f"loss_fn: {hyperparams['enc_loss_fn']}, test_loss_metric: {hyperparams['loss_metric']}")
        print(f"Initial_LR: {hyperparams['enc_lr']}, weight_decay: {hyperparams['enc_weight_decay']}" + "\n")
        ################WandB Logging#################################################################
        wandb.log({"Encoder_Architecture": encoder, "Num_Params": num_params, "Random_SEED": hyperparams['random_seed'],
        "optimizer": hyperparams['enc_optimizer'], "scheduler": hyperparams['enc_scheduler'], 
        "loss_fn": hyperparams['enc_loss_fn'], "test_loss_metric": hyperparams['loss_metric'],
        "Initial_LR": hyperparams['enc_lr'], "weight_decay": hyperparams['enc_weight_decay']})
        ##############################################################################################
        self.dec_optimizer = OPTIMIZERS[hyperparams['dec_optimizer']](params=decoder.parameters(), \
                         lr=hyperparams['dec_lr'], weight_decay=hyperparams['dec_weight_decay'])
        self.dec_scheduler = SCHEDULERS[hyperparams['dec_scheduler']](optimizer=self.dec_optimizer, \
                            max_lr=hyperparams['dec_lr'], total_steps=hyperparams['total_steps'], pct_start=0.2, \
                            div_factor=hyperparams['dec_div_factor'], final_div_factor=hyperparams['dec_final_div_factor'])
        num_dec_params = count_params(decoder)
        self.dec_loss_func  = LOSS[hyperparams['dec_loss_fn']]
        self.dec_input_transform = dec_input_transform
        self.dec_output_transform = dec_output_transform
        ###################Terminal Output############################################################
        print(f'Number of Parameters in the Decoder: {num_dec_params}')
        print(f"decoder optimizer: {hyperparams['dec_optimizer']}, decoder scheduler: {hyperparams['dec_scheduler']}")
        print(f"decoder loss_fn: {hyperparams['dec_loss_fn']}")
        print(f"Decoder Initial_LR: {hyperparams['dec_lr']}, decoder_weight_decay: {hyperparams['dec_weight_decay']}" + "\n")
        ################WandB Logging#################################################################
        wandb.log({"Decoder_Architecture": decoder, "Num_Params": num_dec_params, "Random_SEED": hyperparams['random_seed'],
        "dec_optimizer": hyperparams['dec_optimizer'], "dec_scheduler": hyperparams['dec_scheduler'], 
        "dec_loss_fn": hyperparams['dec_loss_fn'],
        "Dec_Initial_LR": hyperparams['dec_lr'], "dec_weight_decay": hyperparams['dec_weight_decay']})
        ##############################################################################################

    def fit(self, train_dataloader, val_dataloader, test_dataloader):
        best_val_loss = 10000.0
        test_loss1 = 0.0
        test_loss2 = 0.0
        std1 = 0.0
        std2 = 0.0
        epsilon = self.eps
        learning = self.patience
        epoch = 0
        train_start_timer = default_timer()
        while learning and epoch<500:
            epoch_start_timer = default_timer()
            learning -= 1
            epoch += 1
            loss = 0
            validation_loss = 0
            for x, y in train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                loss_batch, pred = self.train(x, y) 
                loss += loss_batch.item()         
                self.enc_scheduler.step()
                self.dec_scheduler.step()
                del x
                del y
                del pred
                del loss_batch
                torch.cuda.empty_cache()  
            with torch.no_grad():
                for x, y in val_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    loss_batch, pred = self.validate(x, y)  
                    validation_loss += loss_batch.item()  
                    # del x
                    # del y
                    # del loss_batch
                    # del pred
                    torch.cuda.empty_cache()            
            loss /= len(train_dataloader)
            validation_loss /= len(val_dataloader)
            #########HACK############################
            if validation_loss < best_val_loss:
                if best_val_loss - validation_loss > epsilon:
                    learning = self.patience
                best_val_loss = validation_loss
                path = f'./models_state_dict/{self.project_name}'
                os.makedirs(path, exist_ok = True) 
                path_enc = path+'/encoder.pt'
                torch.save(self.encoder.state_dict(), path_enc)
                path_dec = path+'/decoder.pt'
                torch.save(self.decoder.state_dict(), path_dec)          
            #########################################
            epoch_time = np.round((default_timer() - epoch_start_timer), 4)
            wandb.log({"Epoch": epoch, "Time": epoch_time,"Train Loss": loss, "Validation Loss": validation_loss})
            print('Epoch := %s || Time (sec):= %s  || Train Loss := %.3e || Validation Loss := %.3e'\
                  %(epoch, epoch_time, loss, validation_loss))
            
        train_time = np.round((default_timer() - train_start_timer), 4)
        print("\n" + "##################################################")
        print(f"Total Train Time (sec): {train_time}")
        wandb.log({"Total_epochs": epoch})
        print("##################################################")
        wandb.finish()

    def fit_evolution(self, train_dataloader, val_dataloader, test_dataloader):
        best_val_loss = 10000.0
        test_loss1 = 0.0
        test_loss2 = 0.0
        std1 = 0.0
        std2 = 0.0
        epsilon = self.eps
        learning = self.patience
        epoch = 0
        train_start_timer = default_timer()
        while learning:
            epoch_start_timer = default_timer()
            learning -= 1
            epoch += 1
            validation_loss = 0
            for x, y in train_dataloader:
                loss = 0
                # pred_f = torch.zeros(y.shape[0],self.res,self.res,y.shape[3]).to(self.device)
                x, y = x.to(self.device), y.to(self.device)
                data_x, data_y = x, y[..., 0]
                # print(data_x.shape)
                for t in range(self.T):
                    loss_batch, pred = self.validate(data_x, data_y) 
                    # print(pred.shape)
                    # pred_f[..., t] = pred.squeeze(-1)
                    loss += loss_batch 
                    if t == self.T - 1:
                        break 
                    data_x, data_y = torch.cat((data_x[..., 1:-self.grid_dim], pred.reshape(pred.shape[0], self.res, self.res, 1), data_x[..., -self.grid_dim:]), dim=-1), y[..., t+1] 
                train_loss = loss.item()
                self.enc_optimizer.zero_grad()
                self.dec_optimizer.zero_grad()
                loss.backward()
                self.enc_optimizer.step()
                self.dec_optimizer.step()
                
                self.enc_scheduler.step()
                self.dec_scheduler.step()
            with torch.no_grad():
                for x, y in val_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    data_x, data_y = x, y[..., 0]
                    for t in range(self.T):
                        loss_batch, pred = self.validate(data_x, data_y)  
                        validation_loss += loss_batch.item() 
                        if t == self.T - 1:
                            break 
                        data_x, data_y = torch.cat((data_x[..., 1:-self.grid_dim], pred.reshape(pred.shape[0], self.res, self.res, 1), data_x[..., -self.grid_dim:]), dim=-1), y[..., t+1] 
            train_loss /= len(train_dataloader)
            validation_loss /= len(val_dataloader)
            #########HACK############################
            if validation_loss < best_val_loss:
                if best_val_loss - validation_loss > epsilon:
                    learning = self.patience
                best_val_loss = validation_loss    
            #########################################
            epoch_time = np.round((default_timer() - epoch_start_timer), 4)
            wandb.log({"Epoch": epoch, "Time": epoch_time,"Train Loss": train_loss, "Validation Loss": validation_loss})
            print('Epoch := %s || Time (sec):= %s  || Train Loss := %.3e || Validation Loss := %.3e'\
                  %(epoch, epoch_time, train_loss, validation_loss))
            
        train_time = np.round((default_timer() - train_start_timer), 4)
        print("\n" + "##################################################")
        print(f"Total Train Time (sec): {train_time}")
        wandb.log({"Total_epochs": epoch})
        print("##################################################")
        wandb.finish()
        path = f'./models_state_dict/{self.project_name}'
        os.makedirs(path, exist_ok = True) 
        path_enc = path+'/encoder.pt'
        torch.save(self.encoder.state_dict(), path_enc)
        path_dec = path+'/decoder.pt'
        torch.save(self.decoder.state_dict(), path_dec)


    def train_oformer(self, x, y):
        self.encoder.train()
        self.decoder.train()
        batch_size = x.shape[0]
        loss_func = self.enc_loss_func(res=self.res)
        if self.input_transform is not None:
            x, y = self.input_transform(x, y)
        pred = self.encoder(*x)
        if self.dec_input_transform is not None:
            dec_in = self.dec_input_transform(pred)
        dec_out = self.decoder(*dec_in)
        if self.output_transform is not None:
            pred = self.output_transform(pred)
        if self.dec_output_transform is not None:
            dec_out = self.dec_output_transform(dec_out)
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()
        loss = loss_func(dec_out, y)
        loss.backward()
        self.enc_optimizer.step()
        self.dec_optimizer.step()
        loss = self.dec_loss_func(dec_out, y)
        return loss, dec_out
    
    def val_oformer(self, x, y):
        self.encoder.eval()
        self.decoder.eval()
        batch_size = x.shape[0]
        loss_func = self.enc_loss_func(res=self.res)
        if self.input_transform is not None:
            x, y = self.input_transform(x, y)
        pred = self.encoder(*x)
        if self.dec_input_transform is not None:
            dec_in = self.dec_input_transform(pred)
        dec_out = self.decoder(*dec_in)
        if self.output_transform is not None:
            pred = self.output_transform(pred)
        if self.dec_output_transform is not None:
            dec_out = self.dec_output_transform(dec_out)
        # loss = self.dec_loss_func(dec_out, y)
        loss = loss_func(dec_out, y)
        return loss, dec_out
    

    def test_oformer(self, test_dataloader):
        start_timer = default_timer()
        loss = 0
        loss2 = 0
        loss_arr = []
        loss2_arr = []
        with torch.no_grad():
            for x, y in test_dataloader:
                batch_size = x.shape[0]
                x, y = x.to(self.device), y.to(self.device)
                if self.input_transform is not None:
                    x, y = self.input_transform(x, y)
                pred = self.encoder(*x)
                if self.dec_input_transform is not None:
                    dec_in = self.dec_input_transform(pred)
                dec_out = self.decoder(*dec_in)
                if self.output_transform is not None:
                    pred = self.output_transform(pred)
                if self.dec_output_transform is not None:
                    dec_out = self.dec_output_transform(dec_out)
                loss_batch = self.dec_loss_func(dec_out, y).item()
                loss += loss_batch
                loss_arr.append(loss_batch)######
                if self.metric_loss_fn is not None:
                    loss_batch = self.metric_loss_fn(dec_out.view(batch_size, -1), y.view(batch_size, -1)).item()
                    loss2 += loss_batch
                    loss2_arr.append(loss_batch)#######
                # del x
                # del y
                # del pred
                # torch.cuda.empty_cache()  
        loss = loss/len(test_dataloader)
        loss2 = loss2/len(test_dataloader)
        end_timer = default_timer()
        return loss, loss2
    
    def test_evl_oformer(self, test_dataloader):
        loss = 0
        loss2 = 0
        loss_arr = []
        loss2_arr = []
        with torch.no_grad():
            for batch in test_dataloader:
                batch[0] = batch[0].to(self.device)
                batch[1] = batch[1].to(self.device)
                x, y = batch[0], batch[1][..., 0]
                batch_size = x.shape[0]
                pred_f = torch.zeros(batch[1].shape[0], self.res*self.res, batch[1].shape[3]).to(self.device)
                for t in range(self.T):
                    x, y = x.to(self.device), y.to(self.device)
                    if self.input_transform is not None:
                        x_, y = self.input_transform(x, y)
                    pred = self.encoder(*x_)
                    if self.dec_input_transform is not None:
                        dec_in = self.dec_input_transform(pred)
                    dec_out = self.decoder(*dec_in)
                    if self.output_transform is not None:
                        pred = self.output_transform(pred)
                    if self.dec_output_transform is not None:
                        dec_out = self.dec_output_transform(dec_out)
                    pred_f[..., t] = dec_out.squeeze(-1)
                    if t == self.T - 1:
                        break
                    x, y = torch.cat((x[..., 1:-self.grid_dim], dec_out.reshape(batch[0].shape[0], self.res, self.res, 1), x[..., -self.grid_dim:]), dim=-1), batch[1][..., t+1] 
                loss_batch = self.dec_loss_func(pred_f.view(pred_f.shape[0], -1), batch[1].view(batch[1].shape[0], -1)).item()
                loss += loss_batch
                loss_arr.append(loss_batch)#####
                if self.metric_loss_fn is not None:
                    loss_batch = self.metric_loss_fn(pred_f.view(pred_f.shape[0], -1), batch[1].view(batch[1].shape[0], -1)).item()
                    loss2 += loss_batch
                    loss2_arr.append(loss_batch)######
                

        loss = loss/len(test_dataloader)
        loss2 = loss2/len(test_dataloader)
        return loss, loss2
    
    
    def train_cgan(self, x, y):
        self.encoder.train()
        self.decoder.train()
        if self.input_transform is not None:
            x = self.input_transform(x)
        pred = self.encoder(x)
        if self.output_transform is not None:
            pred = self.output_transform(pred)
        if self.dec_input_transform is not None:
            dec_in = self.dec_input_transform(pred, x, y)
        dec_out = (self.decoder(dec_in[0]), self.decoder(dec_in[1].detach()))
        if self.dec_output_transform is not None:
            dec_out = self.dec_output_transform(dec_out)
        self.enc_optimizer.zero_grad()
        loss_enc, loss = self.enc_loss_func(pred, dec_out[1], y)
        loss_enc.backward(retain_graph=True)
        self.enc_optimizer.step()
        self.dec_optimizer.zero_grad()
        loss_dec = self.dec_loss_func(dec_out)
        loss_dec.backward()
        self.dec_optimizer.step()      
        return loss, pred


    def val_cgan(self, x, y):
        batch_size = x.shape[0]
        self.encoder.eval()
        if self.input_transform is not None:
            x = self.input_transform(x)
        pred = self.encoder(x)
        if self.output_transform is not None:
            pred = self.output_transform(pred)
        loss = LOSS['RelL2'](pred.view(batch_size, -1), y.view(batch_size, -1))
        return loss, pred

    def test_cgan(self, test_dataloader):
        start_timer = default_timer()
        loss = 0
        loss2 = 0
        loss_arr = []
        loss2_arr = []
        with torch.no_grad():
            for x, y in test_dataloader:
                batch_size = x.shape[0]
                x, y = x.to(self.device), y.to(self.device)
                if self.input_transform is not None:
                    x = self.input_transform(x)
                pred = self.encoder(x)
                if self.output_transform is not None:
                    pred = self.output_transform(pred)
                batch_loss = LOSS['RelL2'](pred.view(batch_size, -1), y.view(batch_size, -1)).item()
                loss += batch_loss
                loss_arr.append(batch_loss) #######
                if self.metric_loss_fn is not None:
                    batch_loss = self.metric_loss_fn(pred.view(batch_size, -1), y.view(batch_size, -1)).item()
                    loss2 += batch_loss
                    loss2_arr.append(batch_loss) #####
        loss = loss/len(test_dataloader)
        loss2 = loss2/len(test_dataloader)
        end_timer = default_timer()

        return loss, loss2, torch.std(torch.tensor(loss_arr), dim=0), torch.std(torch.tensor(loss2_arr), dim=0), (end_timer - start_timer)
    
    def test_evl_cgan(self, test_dataloader):
        loss = 0
        loss2 = 0
        loss_arr = []
        loss2_arr = []
        with torch.no_grad():
            for batch in test_dataloader:
                x, y = batch[0], batch[1][..., 0]
                batch_size = x.shape[0]
                for t in range(self.T):
                    x, y = x.to(self.device), y.to(self.device)
                    if self.input_transform is not None:
                        x = self.input_transform(x)
                    pred = self.encoder(x)
                    if self.output_transform is not None:
                        pred = self.output_transform(pred)
                    batch_loss = LOSS['RelL2'](pred.view(batch_size, -1), y.view(batch_size, -1)).item()
                    loss += batch_loss
                    loss_arr.append(batch_loss) ########
                    if self.metric_loss_fn is not None:
                        loss_batch = self.metric_loss_fn(pred.view(batch_size, -1), y.view(batch_size, -1)).item()
                        loss2 += loss_batch
                        loss2_arr.append(loss_batch) #########
                    x, y = torch.cat((x[..., 1:-self.grid_dim], pred, x[..., -self.grid_dim:]), dim=-1), y[..., t+1] 
        loss = loss/len(test_dataloader)
        loss2 = loss2/len(test_dataloader)
        return loss, loss2
    