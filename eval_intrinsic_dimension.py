import torch
import gpytorch
import yaml
import os
import numpy as np

from models import SVDKL_AE
from utils import load_pickle
from data_loader import DataLoader
from intrinsic_dimension import eval_id


def eval_intrinsic_dimension():
    # Import args
    with open("config.yaml", "r") as file:
        args = yaml.safe_load(file)

    latent_dim = args["latent_dim"]
    h_dim = args["h_dim"]
    grid_size = args["grid_size"]
    training_dataset = args["training_dataset"]
    obs_dim_1 = args["obs_dim_1"]
    obs_dim_2 = args["obs_dim_2"]
    obs_dim_3 = args["obs_dim_3"]

    # Set likelihood
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        latent_dim, rank=0, has_task_noise=True, has_global_noise=False
    )

    # Set model
    model = SVDKL_AE(
        num_dim=latent_dim,
        likelihood=likelihood,
        grid_bounds=(-10.0, 10.0),
        hidden_dim=h_dim,
        grid_size=grid_size,
    )

    # Load data
    directory = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(directory + "/Data", training_dataset)
    data = load_pickle(folder)

    # Load weights
    weights_filename = args["weights_filename"]
    weights_folder = os.path.join(
        directory + "/Results/Pendulum/DKL/Noise_level_0.0/", weights_filename
    )
    model.load_state_dict(torch.load(weights_folder)["model"])
    likelihood.load_state_dict(torch.load(weights_folder)["likelihood"])

    # Load the input into the model
    data_loader = DataLoader(data, obs_dim=(obs_dim_1, obs_dim_2, obs_dim_3))
    input_data = data_loader.get_all_samples()
    input_data = torch.from_numpy(input_data).permute(0, 3, 1, 2)
    mu_hat, var_hat, res, mean, covar, z = model(input_data)

    # Evaluate intrinsic dimension
    ID = eval_id(z.detach().numpy())
    print("Intrinsic dimension: ", ID)


if __name__ == "__main__":
    eval_intrinsic_dimension()
