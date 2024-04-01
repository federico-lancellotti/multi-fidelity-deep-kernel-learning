import torch
import gpytorch
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt

from models import SVDKL_AE, SVDKL_AE_2step
from models import SVDKL_AE_latent_dyn
from utils import load_pickle, plot_latent_dims
from data_loader import DataLoader


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
    ID=3

    # Set likelihood
    likelihood_LF = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        latent_dim, rank=0, has_task_noise=True, has_global_noise=False
    )
    likelihood_fwd_LF = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        latent_dim, rank=0, has_task_noise=True, has_global_noise=False
    )

    likelihood_HF = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        ID, rank=0, has_task_noise=True, has_global_noise=False
    )
    likelihood_fwd_HF = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        ID, rank=0, has_task_noise=True, has_global_noise=False
    )

    # Set models
    model_LF = SVDKL_AE_latent_dyn(
        num_dim=latent_dim,
        likelihood=likelihood_LF,
        likelihood_fwd=likelihood_fwd_LF,
        grid_bounds=(-10.0, 10.0),
        h_dim=h_dim,
        grid_size=grid_size,
        obs_dim=42,
        rho=rho,
    )

    model_LF.eval()
    model_LF.AE_DKL.likelihood.eval()
    model_LF.fwd_model_DKL.likelihood.eval()

    model_HF = SVDKL_AE_latent_dyn(
        num_dim=ID,
        likelihood=likelihood_HF,
        likelihood_fwd=likelihood_fwd_HF,
        grid_bounds=(-10.0, 10.0),
        h_dim=h_dim,
        grid_size=grid_size,
        obs_dim=84,
        rho=rho,
        num_dim_LF=latent_dim,
    )

    model_HF.eval()
    model_HF.AE_DKL.likelihood.eval()
    model_HF.fwd_model_DKL.likelihood.eval()

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
        os.path.join(
            directory + "/Results/Pendulum/DKL/LF", weights_filename[0] + ".pth"
        ),
        os.path.join(
            directory + "/Results/Pendulum/DKL/HF", weights_filename[1] + ".pth"
        ),
    ]

    ## Low fidelity
    model_LF.load_state_dict(torch.load(weights_folder[0])["model"])
    likelihood_LF.load_state_dict(torch.load(weights_folder[0])["likelihood"])
    likelihood_fwd_LF.load_state_dict(torch.load(weights_folder[0])["likelihood_fwd"])

    z_LF = torch.zeros((N[0], latent_dim))
    z_next_LF = torch.zeros((N[0], latent_dim))
    z_fwd_LF = torch.zeros((N[0], latent_dim))
    data_loader_LF = DataLoader(
        data[0], z_LF, z_next_LF, z_fwd_LF, obs_dim=(obs_dim_1[0], obs_dim_2[0], obs_dim_3)
    )

    input_data_LF = data_loader_LF.get_all_samples()
    input_data_LF["obs"] = input_data_LF["obs"].permute(0, 3, 1, 2)
    input_data_LF["next_obs"] = input_data_LF["next_obs"].permute(0, 3, 1, 2)

    mu_x, _, _, _, z_LF, _, mu_next, _, z_next_LF, _, _, _, _, z_fwd_LF = model_LF(input_data_LF["obs"],
                                                                                    input_data_LF["z_LF"],
                                                                                    input_data_LF["next_obs"],
                                                                                    input_data_LF["z_next_LF"],
                                                                                    input_data_LF["z_fwd_LF"],
                                                                                    )
    z_LF = z_LF[:N[1], :].detach()
    z_next_LF = z_next_LF[:N[1], :].detach()
    z_fwd_LF = z_fwd_LF[:N[1], :].detach()

    # High fidelity
    model_HF.load_state_dict(torch.load(weights_folder[1])["model"])
    likelihood_HF.load_state_dict(torch.load(weights_folder[1])["likelihood"])
    likelihood_fwd_HF.load_state_dict(torch.load(weights_folder[1])["likelihood_fwd"])

    data_loader_HF = DataLoader(
        data[1], z_LF, z_next_LF, z_fwd_LF, obs_dim=(obs_dim_1[1], obs_dim_2[1], obs_dim_3)
    )

    input_data_HF = data_loader_HF.get_all_samples()
    input_data_HF["obs"] = input_data_HF["obs"].permute(0, 3, 1, 2)
    input_data_HF["next_obs"] = input_data_HF["next_obs"].permute(0, 3, 1, 2)

    mu_x, _, _, _, z_HF, _, mu_next, _, _, _, _, _, _, _ = model_HF(input_data_HF["obs"],
                                                                input_data_HF["z_LF"],
                                                                input_data_HF["next_obs"],
                                                                input_data_HF["z_next_LF"],
                                                                input_data_HF["z_fwd_LF"],
                                                                )

    # Convert to png
    input_data = data_loader_HF.get_all_samples()["obs"]
    mu_x = mu_x.permute(0, 3, 2, 1)  # move color channel to the end
    mu_x = mu_x.detach().numpy()  # pass to numpy framework
    filepath = directory + "/Results/Pendulum/DKL/plots/" + weights_filename[1] + "/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    plot_latent_dims(z_HF.detach().numpy(), show=False, filename=filepath)

    l = 1
    #for i in np.random.randint(N[l], size=50):
    start = 200*np.random.randint(0,4) + np.random.randint(0,149)
    end = start + 50
    for i in range(start,end):
        mu_x_rec, _, _, _ = model_HF.predict_dynamics_mean(mu_next[i].unsqueeze(dim=0), 
                                                           z_fwd_LF[i])
        mu_x_rec = mu_x_rec.permute(0, 3, 2, 1) # move color channel to the end
        mu_x_rec = mu_x_rec.detach().numpy() # pass to numpy framework

        frame0 = input_data[i, :, :, 0:3] 
        frame1 = mu_x[i, :, :, 0:3]
        frame2 = mu_x_rec[:, :, :, 0:3]    
        
        frame = np.zeros((obs_dim_1[l], obs_dim_2[l]*3, 3), dtype=np.float32)
        frame[:, :obs_dim_2[l], :] = frame0
        frame[:, obs_dim_2[l]:2*obs_dim_2[l], :] = frame1
        frame[:, 2*obs_dim_2[l]:, :] = frame2

        plt.imshow(frame)
        filename = filepath + str(i) + ".png"
        plt.savefig(filename, format="png")


if __name__ == "__main__":
    test()
