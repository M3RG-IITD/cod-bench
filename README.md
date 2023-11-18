# cod-bench
We present cod-bench containing 11 operators and 8 datasets.
# data
The datasets used for benchmarking can be downloaded from here.
https://drive.google.com/drive/folders/1HZ-fHYBVSPHBdl7Lf8wxbuBdh2ilL0dY?usp=share_link


# Using the codebase
4 functionalities are required to take a pair of model and dataset and to perform the training and testing.
1. Load the data
2. Create the Pytorch Model
3. Choose the optimizer, scheduler, loss function and hyperparameter for training.
4. Put them all together, create a training and testing loop to use the model.
   All these functionalities are written using a specific module for each task. More details about each of the module is given below. Completely functional code (Run_files) are attached for more working examples.

# modules
The codebase contains 4 major modules.
1. Loader Modules: Provides functionality of turning datasets to PyTorch DataLoaders. 
2. Operator Modules: Contains Implementation of all models benchamrked in the paper in fully functional form using PyTorch Framework.
3. Utility Modules: Helper functions providing utilities such as transforms, loss functions, optimizers etc.
4. Trainer Module: Provides a common interface for training and testing any pair of model, dataset.

# loader module

Loader Modules:

A. dataloader.py:
   Provides the functionality of processing the raw dataset provided and split it into pytorch tensors of train, validation and test set.
   eg. code:
1. PATH = 'data/Biaxial_N70000_D28.npz'.
2. loader = npzloader(path=PATH).
3. x_train, y_train, x_val, y_val, x_test, y_test = loader.split(ntrain, nval, ntest).


B. loader_1d.py, loader_2d.py:
   Provides the tool to convert the pytorch tensors of data into Pytorch dataloader that can be directly used in the traning process.
   Provides functionality such as input/output normalization, grid_data appended loader etc.
   eg. code:
1. train_obj = DataLoader_2D(X=x_train, y=y_train, n=ntrain, res=training_data_resolution, \
                            grid_size=grid_size, batch_size=batch_size, x_normalizer=x_normalizer)
2. train_grid_loader = train_obj.get_grid_loader()

   

C. loader_gnot.py:
   GNOT combines all data in the form of a graph before training phase. This module allows to create the graph dataloader for all datasets    for use with the GNOT model.

# trainer module
Given model and the choice of optimization and loss functions, hyperparameters, it create a suitable Trainer instance.
Fit function in each of the class of this module trains the model till convergence on the given train_loader.
Test function can be used to see the performance of the model on test set once trained.
eg. code:
1. hyperparameters = {
2.  'lr': 1e-3,
3.  'weight_decay': 1e-4,
4.  'step_size': 100,
5.  'gamma': 0.5,
6.  'optimizer': 'Adam',
7.  'scheduler': 'StepLR',
8.  'loss_fn': 'RelL2',
9.  'loss_metric': 'MSE',
10. 'batch_size': batch_size,
11. 'random_seed': random_seed,
12. }
13. model = FNO2d()
14. trainer = Trainer(model_name=FNO+Biaxial", model=model, hyperparams=hyperparameters, \
                    output_transform=out_transform, device=device)
15. trainer.fit(train_dataloader=train_grid_loader, val_dataloader=val_grid_loader, test_dataloader=test_grid_loader)
16. loss1, loss2 = trainer.test(test_dataloader=test_grid_loader)


# Further Details
1. All code used for experimentation is provided in form of run_{model_name}_{dataset_name}.py python scripts.
2. Since the input output data processing for 5 of the models: FNO, WNO, FNN, ResNet, UNet is similar they all are included in a single file. Such code can be found in scripts like run1_darcy.py, run1_nvs.py etc.
3. The basic data processing is similar for the models: DeepONet, POD-DeepONet, SNO. Code for running these models can be found in files run2_{dataset_name}.py files.
4. Code for running model_name in {'CGAN', 'GNOT', 'OFORMER', 'LSM'} can be found in files named as run_{model_name}_{dataset_name}.py files.

