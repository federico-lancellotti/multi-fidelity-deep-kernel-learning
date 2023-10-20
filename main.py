import torch
import numpy as np
import os
import gpytorch
import gc

from models import SVDKL_AE
from logger import Logger
from utils import load_pickle
from trainer import train
from data_loader import DataLoader

from datetime import datetime

# for 84x84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}

use_gpu = False

# set seed and GPU
seed = 2
torch.manual_seed(seed)
np.random.seed(seed)
if use_gpu: 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('CUDA available:', torch.cuda.is_available())
        torch.cuda.manual_seed(seed)
    elif torch.backends.mps.is_available(): # Apple Silicon
        torch.mps.empty_cache()
        print('MPS available:', torch.backends.mps.is_available())
        torch.mps.manual_seed(seed)

# training hyperparameters
batch_size = 50
max_epoch = 100

training = True
plotting = False
num_samples_plot = 200

# learning rate
lr = 1e-4
lr_gp = 1e-2
lr_gp_var = 1e-2
lr_gp_lik = 1e-2
reg_coef = 1e-2
k1 = 1.0    # coefficient-recon-loss
k2 = 1.0    # coefficient-fwd-kl-loss
grid_size = 32

# build model
latent_dim = 20
act_dim = 1
state_dim = 3
obs_dim_1 = 84
obs_dim_2 = 84
obs_dim_3 = 6
h_dim = 256

# noise level on observations
noise_level = 0.0

# noise level on dynamics (actions)
noise_level_act = 0.0

# experiment and model type
exp = "Pendulum"
mtype = "DKL"
training_dataset = "pendulum_train.pkl"
testing_dataset = "pendulum_test.pkl"

log_interval = 50

jitter = 1e-8


def main(exp='Pendulum', mtype='DKL', noise_level=0.0, training_dataset='pendulum_train.pkl', testing_dataset='pendulum_test.pkl'):
    # load data
    directory = os.path.dirname(os.path.abspath(__file__))

    folder = os.path.join(directory + '/Data', training_dataset)
    folder_test = os.path.join(directory + '/Data', testing_dataset)

    data = load_pickle(folder)
    data_test = load_pickle(folder_test)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(latent_dim, rank=0, has_task_noise=True, has_global_noise=False)

    model = SVDKL_AE(num_dim=latent_dim, likelihood=likelihood, grid_bounds=(-10., 10.), hidden_dim=h_dim, grid_size=grid_size)

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()
            gc.collect()    # NOTE: Critical to avoid GPU leak
        elif torch.backends.mps.is_available(): # on Apple Silicon
            mps_device = torch.device("mps")
            model.to(mps_device)
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr_gp},
        ], lr=lr, weight_decay=reg_coef)
    
    # Set how to save the model
    now = datetime.now()
    date_string = now.strftime("%d-%m-%Y_%Hh-%Mm-%Ss")
    save_pth_dir = directory + '/Results/' + str(exp) + '/' + str(mtype) + '/Noise_level_' + str(noise_level)
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    # Preprocessing of the data
    train_loader = DataLoader(data, obs_dim=(obs_dim_1, obs_dim_2, obs_dim_3))
    test_loader = DataLoader(data_test, obs_dim=(obs_dim_1, obs_dim_2, obs_dim_3))

    # Training
    if training:
        print("start training...")
        for epoch in range(1, max_epoch):
            with gpytorch.settings.cholesky_jitter(jitter):
                train(model=model, train_data=train_loader, optimizer=optimizer, batch_size=batch_size, num_epochs=epoch)

            if epoch % log_interval == 0:
                torch.save({'model': model.state_dict(), 'likelihood': model.likelihood.state_dict()}, 
                           save_pth_dir +'/DKL_Model_' + date_string+'.pth')

        torch.save({'model': model.state_dict(), 'likelihood': model.likelihood.state_dict()}, 
                   save_pth_dir + '/DKL_Model_' + date_string + '.pth')



if __name__ == "__main__":
    main()