import torch
import numpy as np
import os
import gpytorch
import gc

from models import SVDKL_AE, SVDKL_AE_2step
from models import SVDKL_AE_latent_dyn
from variational_inference import VariationalKL
from logger import Logger
from utils import load_pickle, plot_frame
from trainer import train_dyn as train
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
    lr_gp_lik = float(args["lr_gp_lik"])
    lr_gp_var = float(args["lr_gp_var"])
    reg_coef = float(args["reg_coef"])
    k1 = float(args["k1"])
    k2 = float(args["k2"])
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
    likelihood_fwd = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        latent_dim, rank=0, has_task_noise=True, has_global_noise=False
    )

    ####################
    ### low fidelity ###
    ####################

    # Model initialization
    model_LF = SVDKL_AE_latent_dyn(
        num_dim=latent_dim,
        likelihood=likelihood,
        likelihood_fwd=likelihood_fwd,
        grid_bounds=(-10.0, 10.0),
        h_dim=h_dim,
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

    variational_kl_term = VariationalKL(
        model_LF.AE_DKL.likelihood, model_LF.AE_DKL.gp_layer, num_data=batch_size
    )  # len(data)
    variational_kl_term_fwd = VariationalKL(
        model_LF.fwd_model_DKL.likelihood,
        model_LF.fwd_model_DKL.gp_layer_2,
        num_data=batch_size,
    )

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model_LF.AE_DKL.encoder.parameters()},
            {"params": model_LF.AE_DKL.decoder.parameters()},
            {"params": model_LF.fwd_model_DKL.fwd_model.parameters()},
            {"params": model_LF.AE_DKL.gp_layer.hyperparameters(), "lr": lr_gp},
            {
                "params": model_LF.fwd_model_DKL.gp_layer_2.hyperparameters(),
                "lr": lr_gp,
            },
            {"params": model_LF.AE_DKL.likelihood.parameters(), "lr": lr_gp_lik},
            {"params": model_LF.fwd_model_DKL.likelihood.parameters(), "lr": lr_gp_lik},
        ],
        lr=lr,
        weight_decay=reg_coef,
    )

    optimizer_var1 = torch.optim.SGD([
        {'params': model_LF.AE_DKL.gp_layer.variational_parameters(), 'lr': lr_gp_var},
        ], lr=lr_gp_var, momentum=0.9, nesterov=True, weight_decay=0) # momentum=0.9, nesterov=True, weight_decay=0
    optimizer_var2 = torch.optim.SGD([
        {'params': model_LF.fwd_model_DKL.gp_layer_2.variational_parameters(), 'lr': lr_gp_var},
        ], lr=lr_gp_var, momentum=0.9, nesterov=True, weight_decay=0)

    optimizers = [optimizer, optimizer_var1, optimizer_var2]

    # Reduce the learning rate when at 50% and 75% of the training.
    scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_var1, milestones=[0.5 * max_epoch, 0.75 * max_epoch], gamma=0.1
    )
    scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_var2, milestones=[0.5 * max_epoch, 0.75 * max_epoch], gamma=0.1
    )

    # Set how to save the model
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
    save_pth_dir = directory + "/Results/" + str(exp) + "/" + str(mtype) + "/LF/"
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    # Preprocessing of the data
    z_LF = np.zeros((N[0], latent_dim))
    z_next_LF = np.zeros((N[0], latent_dim))
    z_fwd_LF = np.zeros((N[0], latent_dim))
    train_loader_LF = DataLoader(
        data[0],
        z_LF,
        z_next_LF,
        z_fwd_LF,
        obs_dim=(obs_dim_1[0], obs_dim_2[0], obs_dim_3),
    )

    # Training
    if training:
        print("start training...")
        for epoch in range(1, max_epoch + 1):
            with gpytorch.settings.cholesky_jitter(jitter):
                train(
                    model=model_LF,
                    train_data=train_loader_LF,
                    optimizers=optimizers,
                    batch_size=batch_size,
                    num_epochs=epoch,
                    variational_kl_term=variational_kl_term,
                    variational_kl_term_fwd=variational_kl_term_fwd,
                    k1=k1,
                    beta=k2,
                )

                scheduler_1.step()
                scheduler_2.step()

            if epoch % log_interval == 0:
                torch.save(
                    {
                        "model": model_LF.state_dict(),
                        "likelihood": model_LF.likelihood.state_dict(),
                        "likelihood_fwd": model_LF.fwd_model_DKL.likelihood.state_dict(),
                    },
                    save_pth_dir
                    + "/MFDKL_dyn_LF_"
                    + str(obs_dim_1[0])
                    + "_"
                    + date_string
                    + ".pth",
                )

        torch.save(
            {
                "model": model_LF.state_dict(),
                "likelihood": model_LF.AE_DKL.likelihood.state_dict(),
                "likelihood_fwd": model_LF.fwd_model_DKL.likelihood.state_dict(),
            },
            save_pth_dir
            + "/MFDKL_dyn_LF_"
            + str(obs_dim_1[0])
            + "_"
            + date_string
            + ".pth",
        )

    #####################
    ### high fidelity ###
    #####################

    # Low fidelity output
    pred_samples = train_loader_LF.get_all_samples()
    pred_samples["obs"] = torch.from_numpy(pred_samples["obs"]).permute(0, 3, 1, 2)
    pred_samples["next_obs"] = torch.from_numpy(pred_samples["next_obs"]).permute(
        0, 3, 1, 2
    )
    _, _, _, _, z_LF, _, _, _, z_next_LF, _, _, _, _, z_fwd_LF = model_LF(
        pred_samples["obs"],
        pred_samples["z_LF"],
        pred_samples["next_obs"],
        pred_samples["z_next_LF"],
        pred_samples["z_fwd_LF"],
    )
    z_LF = z_LF[0 : N[1]].detach().numpy()
    z_next_LF = z_next_LF[0 : N[1]].detach().numpy()
    z_fwd_LF = np.zeros((N[1], latent_dim))

    # ID estimation
    ID_0 = eval_id(z_LF)
    ID_1 = eval_id(z_next_LF)
    ID_fwd = eval_id(z_fwd_LF)
    print("ID_0=", ID_0, ", ID_1=", ID_1, ", ID_fwd=", ID_fwd)
    # print("ID = ", ID, " ===> ID = ", int(round(ID)))
    ID = int(round(ID_0))

    # Model initialization
    model_HF = SVDKL_AE_latent_dyn(
        num_dim=latent_dim,
        likelihood=likelihood,
        likelihood_fwd=likelihood_fwd,
        grid_bounds=(-10.0, 10.0),
        h_dim=h_dim,
        grid_size=grid_size,
        obs_dim=84,
        rho=rho,
        # ID=ID,
    )

    if use_gpu:
        if torch.cuda.is_available():
            model_HF = model_HF.cuda()
            gc.collect()  # NOTE: Critical to avoid GPU leak
        elif torch.backends.mps.is_available():  # on Apple Silicon
            mps_device = torch.device("mps")
            model_HF.to(mps_device)

    variational_kl_term = VariationalKL(
        model_HF.AE_DKL.likelihood, model_HF.AE_DKL.gp_layer, num_data=batch_size
    )  # len(data)
    variational_kl_term_fwd = VariationalKL(
        model_HF.fwd_model_DKL.likelihood,
        model_HF.fwd_model_DKL.gp_layer_2,
        num_data=batch_size,
    )

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model_HF.AE_DKL.encoder.parameters()},
            {"params": model_HF.AE_DKL.decoder.parameters()},
            {"params": model_HF.fwd_model_DKL.fwd_model.parameters()},
            {"params": model_HF.AE_DKL.gp_layer.hyperparameters(), "lr": lr_gp},
            {"params": model_HF.fwd_model_DKL.gp_layer_2.hyperparameters(), "lr": lr_gp},
            {"params": model_HF.AE_DKL.likelihood.parameters(), "lr": lr_gp_lik},
            {"params": model_HF.fwd_model_DKL.likelihood.parameters(), "lr": lr_gp_lik},
        ],
        lr=lr,
        weight_decay=reg_coef,
    )

    optimizer_var1 = torch.optim.SGD([
        {'params': model_HF.AE_DKL.gp_layer.variational_parameters(), 'lr': lr_gp_var},
        ], lr=lr_gp_var, momentum=0.9, nesterov=True, weight_decay=0) # momentum=0.9, nesterov=True, weight_decay=0
    optimizer_var2 = torch.optim.SGD([
        {'params': model_HF.fwd_model_DKL.gp_layer_2.variational_parameters(), 'lr': lr_gp_var},
        ], lr=lr_gp_var, momentum=0.9, nesterov=True, weight_decay=0)

    optimizers = [optimizer, optimizer_var1, optimizer_var2]

    # Reduce the learning rate when at 50% and 75% of the training.
    scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_var1, milestones=[0.5 * max_epoch, 0.75 * max_epoch], gamma=0.1
    )
    scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_var2, milestones=[0.5 * max_epoch, 0.75 * max_epoch], gamma=0.1
    )

    # Set how to save the model
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
    save_pth_dir = directory + "/Results/" + str(exp) + "/" + str(mtype) + "/HF/"
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    # Preprocessing of the data
    train_loader_HF = DataLoader(
        data[1],
        z_LF,
        z_next_LF,
        z_fwd_LF,
        obs_dim=(obs_dim_1[1], obs_dim_2[1], obs_dim_3),
    )

    # Training
    if training:
        print("start training...")
        for epoch in range(1, max_epoch + 1):
            with gpytorch.settings.cholesky_jitter(jitter):
                train(
                    model=model_HF,
                    train_data=train_loader_HF,
                    optimizers=optimizers,
                    batch_size=batch_size,
                    num_epochs=epoch,
                    variational_kl_term=variational_kl_term,
                    variational_kl_term_fwd=variational_kl_term_fwd,
                    k1=k1,
                    beta=k2,
                )

            if epoch % log_interval == 0:
                torch.save(
                    {
                        "model": model_HF.state_dict(),
                        "likelihood": model_HF.AE_DKL.likelihood.state_dict(),
                        "likelihood_fwd": model_HF.fwd_model_DKL.likelihood.state_dict(),
                    },
                    save_pth_dir
                    + "/MFDKL_dyn_HF_"
                    + str(obs_dim_1[0])
                    + "-"
                    + str(obs_dim_1[1])
                    + "_"
                    # + "ID="
                    # + str(ID)
                    # + "_"
                    + date_string + ".pth",
                )

        torch.save(
            {
                "model": model_HF.state_dict(),
                "likelihood": model_HF.AE_DKL.likelihood.state_dict(),
                "likelihood_fwd": model_HF.fwd_model_DKL.likelihood.state_dict(),
            },
            save_pth_dir
            + "/MFDKL_dyn_HF_"
            + str(obs_dim_1[0])
            + "-"
            + str(obs_dim_1[1])
            + "_"
            # + "ID="
            # + str(ID)
            # + "_"
            + date_string + ".pth",
        )

    print("##############################################")
    print("ID = ", ID)


if __name__ == "__main__":
    main()
