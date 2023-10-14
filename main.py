import torch
import numpy as np
import os
import gpytorch
import gc

from models import SVDKL_AE
from logger import Logger
from utils import load_pickle


# for 84x84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}

# set seed and GPU
seed = 2
torch.manual_seed(seed)
np.random.seed(seed)
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

    model = SVDKL_AE(num_dim=latent_dim, likelihood=likelihood, grid_bounds=(-10.0, 10.0), hidden_dim=h_dim, grid_size=grid_size)

    if torch.cuda.is_available():
        model = model.cuda()
        variational_kl_term = variational_kl_term.cuda()
        variational_kl_term_fwd = variational_kl_term_fwd.cuda()
        gc.collect()    # NOTE: Critical to avoid GPU leak
    elif torch.backends.mps.is_available(): # on Apple Silicon
        mps_device = torch.device("mps")
        model.to(mps_device)