import torch
import gpytorch
import yaml
import os

from models import MF_SVDKL_AE
from models import MF_SVDKL_AE_2step
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
    test_set = args["testing_dataset"]
    obs_dim_1 = args["obs_dim_1"]
    obs_dim_2 = args["obs_dim_2"]
    obs_dim_3 = args["obs_dim_3"]
    rho = args["rho"]
    batch_size = args["batch_size"]

    # Set likelihood
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        latent_dim, rank=0, has_task_noise=True, has_global_noise=False
    )

    # Set model
    model = MF_SVDKL_AE(
        num_dim=latent_dim,
        likelihood=likelihood,
        grid_bounds=(-10.0, 10.0),
        hidden_dim=h_dim,
        grid_size=grid_size,
        rho=rho,
    )

    # Load data
    directory = os.path.dirname(os.path.abspath(__file__))
    folder = [
        os.path.join(directory + "/Data", test_set[0]),
        os.path.join(directory + "/Data", test_set[1]),
    ]
    data = [load_pickle(folder[0]), load_pickle(folder[1])]

    # Load weights
    weights_filename = args["weights_filename"]
    weights_folder = os.path.join(
        directory + "/Results/Pendulum/DKL/", weights_filename + ".pth"
    )
    model.load_state_dict(torch.load(weights_folder)["model"])
    likelihood.load_state_dict(torch.load(weights_folder)["likelihood"])

    # Passare i dati di input attraverso il modello
    data_loader = [
        DataLoader(data[0], obs_dim=(obs_dim_1[0], obs_dim_2[0], obs_dim_3)),
        DataLoader(data[1], obs_dim=(obs_dim_1[1], obs_dim_2[1], obs_dim_3)),
    ]

    input_data = [data_loader[0].sample_batch(batch_size), data_loader[1].sample_batch(batch_size)]

    input_data[0] = torch.from_numpy(input_data[0]).permute(0, 3, 1, 2)
    input_data[1] = torch.from_numpy(input_data[1]).permute(0, 3, 1, 2)

    _, _, mu_x_HF, var_x_HF, _, _, _, z = model(input_data[0], input_data[1])
    # mu_x_HF, z = model(input_data[0], input_data[1])

    # Convert to png
    mu_x_HF = mu_x_HF.permute(0, 3, 2, 1)  # move color channel to the end
    mu_x_HF = mu_x_HF.detach().numpy()  # pass to numpy framework
    filepath = directory + "/Results/Pendulum/DKL/plots/" + weights_filename + "/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    for i in range(batch_size):
        frame1 = mu_x_HF[i,:,:,0:3]
        filename1 = filepath + str(i) + str("_1") + ".png"
        plt.imshow(frame1)  
        plt.savefig(filename1, format="png")

        frame2 = mu_x_HF[i,:,:,3:6]
        filename2 = filepath + str(i) + str("_2") + ".png"
        plt.imshow(frame2)  
        plt.savefig(filename2, format="png")


if __name__ == "__main__":
    test()
