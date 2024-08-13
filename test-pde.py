import yaml
import os

from src.utils import plot_latent_dims, plot_error
from src.BuildModel import BuildModel
from test_utils import reconstruct, extrapolate


def test():
    # Import args
    with open("config.yaml", "r") as file:
        args = yaml.safe_load(file)

    latent_dim = args["latent_dim"]
    dt = args["dt"]
    T_test = args["T_test"]
    obs_dim_1 = args["obs_dim_1"]
    obs_dim_2 = args["obs_dim_2"]
    mu_test = args["mu_test"]
    env_name = args["env_name"]
    results_folder = args["results_folder"]
    ID = args["ID"]
    episodes = len(mu_test)
    L_test = int(T_test / dt - 2)
    T_start = args["T"][1]
    N_iter = 10    # Number of iterations for the extrapolation (< L_test, for correct computation of the error)

    print("Testing the model...")

    MF_DKL = BuildModel(args, test=True)
    N = MF_DKL.N

    model_0 = MF_DKL.add_level(level=0, latent_dim=latent_dim)
    model_1 = MF_DKL.add_level(level=1, latent_dim=ID, latent_dim_LF=latent_dim)

    ## Level 0
    model_0, data_loader_0 = MF_DKL.test_level(level=0, model=model_0)
    z_0, _, z_next_0, z_fwd_0, _, _ = MF_DKL.eval_level(model=model_0, data_loader=data_loader_0)

    ## Level 1
    z_LF = z_0[0 : N[1]].detach()
    z_next_LF = z_next_0[0 : N[1]].detach()
    z_fwd_LF = z_fwd_0[0 : N[1]].detach()

    model_1, data_loader_1 = MF_DKL.test_level(level=1, model=model_1, z_LF=z_LF, z_next_LF=z_next_LF, z_fwd_LF=z_fwd_LF)
    z_1, var_1, z_next_1, z_fwd_1, mu_x_1, mu_next_1 = MF_DKL.eval_level(model=model_1, data_loader=data_loader_1)

    # Set the submodel to evaluate
    l = 1
    model = model_1
    data_loader = data_loader_1
    mu_x = mu_x_1
    mu_next = mu_next_1
    z = z_1
    var = var_1
    z_fwd_LF = z_fwd_0.detach()

    # Preprocess reconstruction for the plots
    input_data = data_loader.get_all_samples()["obs"]
    mu_x = mu_x.permute(0, 2, 3, 1)  # move color channel to the end
    mu_x = mu_x.detach().numpy()  # pass to numpy framework
    filepath = MF_DKL.directory + "/Results/" + env_name + "/" + results_folder + "/plots/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Plot of the latent variables
    if not os.path.exists(filepath+"latent_space/"):
        os.makedirs(filepath+"latent_space/")
    plot_latent_dims(z.detach().numpy(), var.detach().numpy(), T_start=T_start, T=L_test, dt=dt, dims=ID, episodes=episodes, show=False, filename=filepath+"latent_space/")
    
    # Plot of reconstructions, predictions and extrapolation
    for index in range(len(mu_test)):
        mu_value = mu_test[index]
        print("Exporting reconstructions for mu = ", mu_value, "...")
        
        filepath_mu = filepath + "mu_" + str(mu_value) + "/"
        if not os.path.exists(filepath_mu):
            os.makedirs(filepath_mu)

        # Plot of the reconstruction
        start = index * L_test
        end = start + L_test - 2
        
        mse, mse_next = reconstruct(model, input_data, mu_x, mu_next, z_fwd_LF, L_test, T_start, dt, start, end, mu_value, filepath_mu)
        
        ## Plot the error
        plot_error([mse, mse_next], T_start, dt, ["MSE", "MSE_next"], filepath_mu)

        # Extrapolation
        mse_extra = extrapolate(model, input_data, mu_next[start], z_fwd_LF, L_test, T_start, dt, filepath_mu, N_iter=N_iter, start=start)

        ## Plot the extrapolation error
        plot_error([mse_extra], T_start, dt, ["MSE (extrapolation)"], filepath_mu + "extrapolation/")


if __name__ == "__main__":
    test()
