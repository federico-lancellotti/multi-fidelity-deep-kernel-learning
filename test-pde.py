import yaml
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import plot_latent_dims, plot_error
from src.BuildModel import BuildModel

def plot_reconstruction(real_frames, images, i, T_start, dt, mu, filepath):
    t = T_start + round(i*dt, 2)
    t_next = T_start + round((i+1)*dt, 2)

    # Generate a grid of 1x4 subplots
    fig, axs = plt.subplots(1, 4, figsize=(10, 10))

    # Compute MSE
    mse = np.mean((images[0] - real_frames[0]) ** 2)
    mse_next = np.mean((images[2] - real_frames[1]) ** 2)

    # First subplot
    axs[0].imshow(images[0])
    axs[0].set_title('t = ' + str(t))

    # Second subplot
    axs[1].imshow(images[1])
    axs[1].set_title('(MSE = ' + str(round(mse, 4)) + ')')

    # Third subplot
    axs[2].imshow(images[2])
    axs[2].set_title('t+dt = ' + str(t_next))

    # Fourth subplot
    axs[3].imshow(images[3])
    axs[3].set_title('(MSE = ' + str(round(mse_next, 4)) + ')')

    # Add a title
    # fig.suptitle(r'$\mu$ = ' + str(mu), fontsize=16)
    # fig.text(0.25, 0.95, 'Reconstruction', ha='center', va='center', fontsize=14)
    # fig.text(0.75, 0.95, 'Error', ha='center', va='center', fontsize=14)

    # Save the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    filename = filepath + str(i) + ".svg"
    plt.savefig(filename, format="svg")
    plt.close()

    return mse, mse_next


def reconstruct(model, input_data, mu_x, mu_next, z_fwd_LF, L_test, T_start, dt, start, end, mu_value, filepath):
    mse = []
    mse_next = []

    for i in range(start,end):
        mu_x_rec, _, _, _ = model.predict_dynamics_mean(mu_next[i].unsqueeze(dim=0), z_fwd_LF[i])
        mu_x_rec = mu_x_rec.permute(0, 2, 3, 1) # move color channel to the end

        # Set the images
        frame = input_data[i, :, :, 0:3].detach().numpy()
        frame_next = input_data[i+1, :, :, 0:3].detach().numpy()
        image1 = mu_x[i, :, :, 0:3]
        image2 = 1 - np.abs(image1 - frame)
        image3 = mu_x_rec[0, :, :, 0:3].detach().numpy()
        image4 = 1 - np.abs(image3 - frame_next)

        mse_new, mse_next_new = plot_reconstruction([frame, frame_next], 
                                                    [image1, image2, image3, image4], 
                                                    i%L_test, T_start, dt, mu_value, filepath)
        mse.append(mse_new)
        mse_next.append(mse_next_new)

    return mse, mse_next


def extrapolate(model, input_data, mu_fwd_init, z_fwd_LF, L_test, T_start, dt, filepath, N_iter=20, start=0):
    mse_extra = []
    mu_fwd = mu_fwd_init.unsqueeze(dim=0)
    if not os.path.exists(filepath + "extrapolation/"):
        os.makedirs(filepath + "extrapolation/")
    
    for i in range(start, start+N_iter):
        mu_x_rec, _, mu_fwd, _ = model.predict_dynamics_mean(mu_fwd, z_fwd_LF[i])
        mu_x_rec = mu_x_rec.permute(0, 2, 3, 1)
        mu_x_rec = mu_x_rec.detach().numpy()

        frame = input_data[i+1, :, :, 0:3].detach().numpy()
        image1 = mu_x_rec[:, :, :, 0:3].squeeze(0)
        image2 = 1 - np.abs(image1 - frame)

        # Generate a grid of 1x2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(6, 4))
        t = T_start + round((i%L_test)*dt, 2)

        # Compute MSE
        mse = np.mean((image1 - frame) ** 2)
        mse_extra.append(mse)

        # First subplot
        axs[0].imshow(image1)
        axs[0].set_title('t = ' + str(t))

        # Second subplot
        axs[1].imshow(image2)
        axs[1].set_title('(MSE = ' + str(round(mse, 4)) + ')')

        # Save the image
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        filename = filepath + "extrapolation/" + str(i%L_test) + ".svg"
        plt.savefig(filename, format="svg")
        plt.close()

    return mse_extra


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
