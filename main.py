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

import yaml

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
    seed = args["seed"]
    batch_size = args["batch_size"]
    max_epoch = args["max_epoch"]
    training = args["training"]
    # plotting = args["plotting"]
    # num_samples_plot = args["num_samples_plot"]
    lr = float(args["lr"])
    lr_gp = float(args["lr_gp"])
    # lr_gp_var = args["lr_gp_var"]
    # lr_gp_lik = args["lr_gp_lik"]
    reg_coef = float(args["reg_coef"])
    # k1 = args["k1"]
    # k2 = args["k2"]
    grid_size = args["grid_size"]
    latent_dim = args["latent_dim"]
    # act_dim = args["act_dim"]
    # state_dim = args["state_dim"]
    obs_dim_1 = args["obs_dim_1"]
    obs_dim_2 = args["obs_dim_2"]
    obs_dim_3 = args["obs_dim_3"]
    h_dim = args["h_dim"]
    noise_level = args["noise_level"]
    # noise_level_act = args["noise_level_act"]
    exp = args["exp"]
    mtype = args["mtype"]
    training_dataset = args["training_dataset"]
    testing_dataset = args["testing_dataset"]
    log_interval = args["log_interval"]
    jitter = float(args["jitter"])

    # Load data
    directory = os.path.dirname(os.path.abspath(__file__))

    folder = os.path.join(directory + "/Data", training_dataset)
    folder_test = os.path.join(directory + "/Data", testing_dataset)

    data = load_pickle(folder)
    data_test = load_pickle(folder_test)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        latent_dim, rank=0, has_task_noise=True, has_global_noise=False
    )

    # Model initialization
    model = SVDKL_AE(
        num_dim=latent_dim,
        likelihood=likelihood,
        grid_bounds=(-10.0, 10.0),
        hidden_dim=h_dim,
        grid_size=grid_size,
    )

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()
            gc.collect()  # NOTE: Critical to avoid GPU leak
        elif torch.backends.mps.is_available():  # on Apple Silicon
            mps_device = torch.device("mps")
            model.to(mps_device)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters()},
            {"params": model.decoder.parameters()},
            {"params": model.gp_layer.hyperparameters(), "lr": lr_gp},
        ],
        lr=lr,
        weight_decay=reg_coef,
    )

    # Set how to save the model
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
    save_pth_dir = (
        directory
        + "/Results/"
        + str(exp)
        + "/"
        + str(mtype)
        + "/Noise_level_"
        + str(noise_level)
    )
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    # Preprocessing of the data
    train_loader = DataLoader(data, obs_dim=(obs_dim_1, obs_dim_2, obs_dim_3))
    test_loader = DataLoader(data_test, obs_dim=(obs_dim_1, obs_dim_2, obs_dim_3))

    # Training
    if training:
        print("start training...")
        for epoch in range(1, max_epoch + 1):
            with gpytorch.settings.cholesky_jitter(jitter):
                train(
                    model=model,
                    train_data=train_loader,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    num_epochs=epoch,
                )

            if epoch % log_interval == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "likelihood": model.likelihood.state_dict(),
                    },
                    save_pth_dir
                    + "/DKL_Model_"
                    + str(obs_dim_1)
                    + "_"
                    + date_string
                    + ".pth",
                )

        torch.save(
            {"model": model.state_dict(), "likelihood": model.likelihood.state_dict()},
            save_pth_dir + "/DKL_Model_" + str(obs_dim_1) + "_" + date_string + ".pth",
        )


if __name__ == "__main__":
    main()
