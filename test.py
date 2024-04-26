import yaml
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import load_pickle, plot_latent_dims
from BuildModel import BuildModel


def test():
    # Import args
    with open("config.yaml", "r") as file:
        args = yaml.safe_load(file)

    latent_dim = args["latent_dim"]
    obs_dim_1 = args["obs_dim_1"]
    obs_dim_2 = args["obs_dim_2"]
    exp = args["exp"]
    results_folder = args["results_folder"]
    ID = args["ID"]

    MF_DKL = BuildModel(args, test=True)
    N = MF_DKL.N


    model_0 = MF_DKL.add_level(level=0, latent_dim=latent_dim)
    model_1 = MF_DKL.add_level(level=1, latent_dim=latent_dim)
    model_2 = MF_DKL.add_level(level=2, latent_dim=latent_dim)
    model_3 = MF_DKL.add_level(level=3, latent_dim=ID, latent_dim_LF=latent_dim)

    ## Level 0
    model_0, data_loader_0 = MF_DKL.test_level(level=0, model=model_0)
    z_0, z_next_0, z_fwd_0, _, _ = MF_DKL.eval_level(model=model_0, data_loader=data_loader_0)

    ## Level 1
    model_1, data_loader_1 = MF_DKL.test_level(level=1, model=model_1)
    z_1, z_next_1, z_fwd_1, _, _ = MF_DKL.eval_level(model=model_1, data_loader=data_loader_1)

    ## Level 2
    model_2, data_loader_2 = MF_DKL.test_level(level=2, model=model_2)
    z_2, z_next_2, z_fwd_2, _, _ = MF_DKL.eval_level(model=model_2, data_loader=data_loader_2)

    # Level 3
    z_LF = z_0[0 : N[3]].detach() + z_1[0 : N[3]].detach() + z_2[0 : N[3]].detach()
    z_next_LF = z_next_0[0 : N[3]].detach() + z_next_1[0 : N[3]].detach() + z_next_2[0 : N[3]].detach()
    z_fwd_LF = z_fwd_0[0 : N[3]].detach() + z_fwd_1[0 : N[3]].detach() + z_fwd_2[0 : N[3]].detach()

    model_3, data_loader_3 = MF_DKL.test_level(level=3, model=model_3, z_LF=z_LF, z_next_LF=z_next_LF, z_fwd_LF=z_fwd_LF)
    z_3, _, _, mu_x, mu_next = MF_DKL.eval_level(model=model_3, data_loader=data_loader_3)


    # Preprocess reconstruction for the plots
    input_data = data_loader_3.get_all_samples()["obs"]
    mu_x = mu_x.permute(0, 2, 3, 1)  # move color channel to the end
    mu_x = mu_x.detach().numpy()  # pass to numpy framework
    filepath = MF_DKL.directory + "/Results/" + exp + "/" + results_folder + "/plots/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Plot of the latent variables
    plot_latent_dims(z_3.detach().numpy(), dims=ID, episodes=3, show=False, filename=filepath)

    # Plot of the reconstruction
    l = 3   # level chosen
    #for i in np.random.randint(N[l], size=50):
    start = 200*np.random.randint(0,4) + np.random.randint(0,149)
    end = start + 50
    for i in range(start,end):
        mu_x_rec, _, _, _ = model_3.predict_dynamics_mean(mu_next[i].unsqueeze(dim=0), z_fwd_LF[i])
        mu_x_rec = mu_x_rec.permute(0, 2, 3, 1) # move color channel to the end
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
