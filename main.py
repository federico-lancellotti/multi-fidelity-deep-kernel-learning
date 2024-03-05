import torch
import numpy as np
import os
import gpytorch
import gc

from models import SVDKL_AE
from models import SVDKL_AE_2step
from logger import Logger
from utils import load_pickle
from trainer import train
from data_loader import DataLoader
from intrinsic_dimension import eval_id

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
    rho = args["rho"]
    training = args["training"]
    lr = float(args["lr"])
    lr_gp = float(args["lr_gp"])
    reg_coef = float(args["reg_coef"])
    grid_size = args["grid_size"]
    latent_dim = args["latent_dim"]
    obs_dim_1 = args["obs_dim_1"]
    obs_dim_2 = args["obs_dim_2"]
    obs_dim_3 = args["obs_dim_3"]
    h_dim = args["h_dim"]
    exp = args["exp"]
    mtype = args["mtype"]
    training_dataset = args["training_dataset"]
    log_interval = args["log_interval"]
    jitter = float(args["jitter"])

    # Load data
    directory = os.path.dirname(os.path.abspath(__file__))

    folder = [
        os.path.join(directory + "/Data", training_dataset[0]),
        os.path.join(directory + "/Data", training_dataset[1]),
    ]

    data = [load_pickle(folder[0]), load_pickle(folder[1])]
    N = [len(data[0]), len(data[1])]

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        latent_dim, rank=0, has_task_noise=True, has_global_noise=False
    )

    ####################
    ### low fidelity ###
    ####################

    # Model initialization
    model_LF = SVDKL_AE(
        num_dim=latent_dim,
        likelihood=likelihood,
        grid_bounds=(-10.0, 10.0),
        hidden_dim=h_dim,
        grid_size=grid_size,
        obs_dim=42,
        rho=rho,
    )

    if use_gpu:
        if torch.cuda.is_available():
            model_LF = model_LF.cuda()
            gc.collect()  # NOTE: Critical to avoid GPU leak
        elif torch.backends.mps.is_available():  # on Apple Silicon
            mps_device = torch.device("mps")
            model_LF.to(mps_device)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model_LF.encoder.parameters()},
            {"params": model_LF.decoder.parameters()},
            {"params": model_LF.gp_layer.hyperparameters(), "lr": lr_gp},
        ],
        lr=lr,
        weight_decay=reg_coef,
    )

    # Set how to save the model
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
    save_pth_dir = directory + "/Results/" + str(exp) + "/" + str(mtype) + "/LF/"
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    # Preprocessing of the data
    z_LF = np.zeros((N[0], latent_dim))
    train_loader_LF = DataLoader(
        data[0], z_LF, obs_dim=(obs_dim_1[0], obs_dim_2[0], obs_dim_3)
    )

    # Training
    if training:
        print("start training...")
        for epoch in range(1, max_epoch + 1):
            with gpytorch.settings.cholesky_jitter(jitter):
                train(
                    model=model_LF,
                    train_data=train_loader_LF,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    num_epochs=epoch,
                )

            if epoch % log_interval == 0:
                torch.save(
                    {
                        "model": model_LF.state_dict(),
                        "likelihood": model_LF.likelihood.state_dict(),
                    },
                    save_pth_dir
                    + "/MFDKL_LF_"
                    + str(obs_dim_1[0])
                    + "_"
                    + date_string
                    + ".pth",
                )

        torch.save(
            {
                "model": model_LF.state_dict(),
                "likelihood": model_LF.likelihood.state_dict(),
            },
            save_pth_dir
            + "/MFDKL_LF_"
            + str(obs_dim_1[0])
            + "_"
            + date_string
            + ".pth",
        )

    #####################
    ### high fidelity ###
    #####################

    # Low fidelity output
    pred_data, z_LF = train_loader_LF.get_all_samples()
    pred_data = torch.from_numpy(pred_data).permute(0, 3, 1, 2)
    _, _, _, _, _, z_LF = model_LF(pred_data, z_LF)
    z_LF = z_LF[0:N[1]].detach().numpy()
    # z_LF = np.zeros((N[1], latent_dim))
    ID = int(np.ceil(eval_id(z_LF)))

    # Model initialization
    model_HF = SVDKL_AE_2step(
        num_dim=latent_dim,
        likelihood=likelihood,
        grid_bounds=(-10.0, 10.0),
        hidden_dim=h_dim,
        grid_size=grid_size,
        obs_dim=84,
        rho=rho,
        ID=ID,
    )

    if use_gpu:
        if torch.cuda.is_available():
            model_HF = model_HF.cuda()
            gc.collect()  # NOTE: Critical to avoid GPU leak
        elif torch.backends.mps.is_available():  # on Apple Silicon
            mps_device = torch.device("mps")
            model_HF.to(mps_device)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model_HF.ext_encoder.parameters()},
            {"params": model_HF.ext_decoder.parameters()},
            {"params": model_HF.gp_layer.hyperparameters(), "lr": lr_gp},
            {"params": model_HF.int_encoder.parameters()},
            {"params": model_HF.int_decoder.parameters()},
        ],
        lr=lr,
        weight_decay=reg_coef,
    )

    # Set how to save the model
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
    save_pth_dir = directory + "/Results/" + str(exp) + "/" + str(mtype) + "/HF/"
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    # Preprocessing of the data
    train_loader_HF = DataLoader(
        data[1], z_LF, obs_dim=(obs_dim_1[1], obs_dim_2[1], obs_dim_3)
    )

    # Training
    if training:
        print("start training...")
        for epoch in range(1, max_epoch + 1):
            with gpytorch.settings.cholesky_jitter(jitter):
                train(
                    model=model_HF,
                    train_data=train_loader_HF,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    num_epochs=epoch,
                )

            if epoch % log_interval == 0:
                torch.save(
                    {
                        "model": model_HF.state_dict(),
                        "likelihood": model_HF.likelihood.state_dict(),
                    },
                    save_pth_dir
                    + "/MFDKL_HF_"
                    + str(obs_dim_1[0])
                    + "-"
                    + str(obs_dim_1[1])
                    + "_"
                    + date_string
                    + ".pth",
                )

        torch.save(
            {
                "model": model_HF.state_dict(),
                "likelihood": model_HF.likelihood.state_dict(),
            },
            save_pth_dir
            + "/MFDKL_HF_"
            + str(obs_dim_1[0])
            + "-"
            + str(obs_dim_1[1])
            + "_"
            + date_string
            + ".pth",
        )


if __name__ == "__main__":
    main()
