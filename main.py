import torch
import numpy as np
import os
import gpytorch
import gc
from datetime import datetime
import yaml

from models import SVDKL_AE, SVDKL_AE_2step
from models import SVDKL_AE_latent_dyn
from variational_inference import VariationalKL
from logger import Logger
from utils import load_pickle, plot_frame
from trainer import train_dyn as train
from data_loader import DataLoader
from intrinsic_dimension import eval_id
from BuildModel import BuildModel


# for 84x84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}

use_gpu = False


def main():
    # Import args
    with open("config.yaml", "r") as file:
        args = yaml.safe_load(file)

    # Set seed and GPU
    seed = args["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_gpu:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA available:", torch.cuda.is_available())
            torch.cuda.manual_seed(seed)
        elif torch.backends.mps.is_available():  # Apple Silicon
            torch.mps.empty_cache()
            print("MPS available:", torch.backends.mps.is_available())
            torch.mps.manual_seed(seed)

    # Set parameters
    latent_dim = args['latent_dim']

    # Build the model
    MF_DKL = BuildModel(args)
    N = MF_DKL.N

    # LEVEL OF FIDELITY: 0
    model_0, train_loader_0 = MF_DKL.add_level(level=0, latent_dim=latent_dim)
    z_0, z_next_0, z_fwd_0 = MF_DKL.eval_level(model_0, train_loader_0)

    z_0 = z_0[0 : N[1]].detach()
    z_next_0 = z_next_0[0 : N[1]].detach()
    z_fwd_0 = z_fwd_0[0 : N[1]].detach()

    # ID estimation
    ID_0 = eval_id(z_0)
    ID_1 = eval_id(z_next_0)
    ID_fwd = eval_id(z_fwd_0)
    print("ID_0=", ID_0, ", ID_1=", ID_1, ", ID_fwd=", ID_fwd)
    # print("ID = ", ID, " ===> ID = ", int(round(ID)))
    ID = int(round((ID_0 + ID_1)/2 + ID_fwd))

    # LEVEL OF FIDELITY: 1
    model_1 = MF_DKL.add_level(level=1, latent_dim=ID, z_LF=z_0, z_next_LF=z_next_0, z_fwd_LF=z_fwd_0, latent_dim_LF=latent_dim)

    print("##############################################")
    print("ID = ", ID)


if __name__ == "__main__":
    main()
