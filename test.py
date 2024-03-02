import torch
import gpytorch
import yaml
import os
import numpy as np

from models import SVDKL_AE
from utils import load_pickle
from data_loader import DataLoader
import matplotlib.pyplot as plt


def test():
    # Import args
    with open("config.yaml", "r") as file:
        args = yaml.safe_load(file)

    latent_dim = args["latent_dim"]
    h_dim = args["h_dim"]
    grid_size = args["grid_size"]
    test_set = args["training_dataset"]
    #test_set = args["testing_dataset"]
    obs_dim_1 = args["obs_dim_1"]
    obs_dim_2 = args["obs_dim_2"]
    obs_dim_3 = args["obs_dim_3"]
    rho = args["rho"]
    batch_size = args["batch_size"]

    # Set likelihood
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        latent_dim, rank=0, has_task_noise=True, has_global_noise=False
    )

    # Set models
    model_LF = SVDKL_AE(
        num_dim=latent_dim,
        likelihood=likelihood,
        grid_bounds=(-10.0, 10.0),
        hidden_dim=h_dim,
        grid_size=grid_size,
        obs_dim=42,
        rho=rho,
    )

    model_HF = SVDKL_AE(
        num_dim=latent_dim,
        likelihood=likelihood,
        grid_bounds=(-10.0, 10.0),
        hidden_dim=h_dim,
        grid_size=grid_size,
        obs_dim=84,
        rho=rho,
    )

    # Load data
    directory = os.path.dirname(os.path.abspath(__file__))
    folder = [
        os.path.join(directory + "/Data", test_set[0]),
        os.path.join(directory + "/Data", test_set[1]),
    ]
    data = [load_pickle(folder[0]), load_pickle(folder[1])]
    N = [len(data[0]), len(data[1])]

    # Load weights
    weights_filename = args["weights_filename"]
    weights_folder = [
        os.path.join(directory + "/Results/Pendulum/LF/DKL/", weights_filename[0] + ".pth"),
        os.path.join(directory + "/Results/Pendulum/HF/DKL/", weights_filename[1] + ".pth")]

    ## Low fidelity
    model_LF.load_state_dict(torch.load(weights_folder[0])["model"])
    likelihood.load_state_dict(torch.load(weights_folder[0])["likelihood"])

    z_LF = np.zeros((N[0], latent_dim))
    data_loader_LF = DataLoader(data[0], z_LF, obs_dim=(obs_dim_1[0], obs_dim_2[0], obs_dim_3))

    input_data_LF, z_LF = data_loader_LF.get_all_samples()
    input_data_LF = torch.from_numpy(input_data_LF).permute(0, 3, 1, 2)
    
    mu_hat, _, _, _, _, z_LF = model_LF(input_data_LF, z_LF)
    z_LF = z_LF.detach().numpy()

    ## High fidelity
    model_HF.load_state_dict(torch.load(weights_folder[1])["model"])
    likelihood.load_state_dict(torch.load(weights_folder[1])["likelihood"])

    z_LF = np.zeros((N[1], latent_dim))
    data_loader_HF = DataLoader(data[1], z_LF[1], obs_dim=(obs_dim_1[1], obs_dim_2[1], obs_dim_3))

    input_data_HF, z_LF = data_loader_HF.get_all_samples()
    input_data_HF = torch.from_numpy(input_data_HF).permute(0, 3, 1, 2)
    
    mu_hat, var_hat, res, mean, covar, z = model_HF(input_data_HF, z_LF)

    # Convert to png
    mu_hat = mu_hat.permute(0, 3, 2, 1)  # move color channel to the end
    mu_hat = mu_hat.detach().numpy()  # pass to numpy framework
    filepath = directory + "/Results/Pendulum/DKL/plots/" + weights_filename[1] + "/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    for i in range(50):
        frame1 = mu_hat[i,:,:,0:3]
        filename1 = filepath + str(i) + str("_1") + ".png"
        plt.imshow(frame1)  
        plt.savefig(filename1, format="png")

        frame2 = mu_hat[i,:,:,3:6]
        filename2 = filepath + str(i) + str("_2") + ".png"
        plt.imshow(frame2)  
        plt.savefig(filename2, format="png")


if __name__ == "__main__":
    test()
