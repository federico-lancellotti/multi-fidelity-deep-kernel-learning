import torch
import numpy as np
import yaml

from intrinsic_dimension import estimate_ID
from BuildModel import BuildModel


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
    latent_dim = args['latent_dim']

    # Build the model
    MF_DKL = BuildModel(args, use_gpu=use_gpu)
    N = MF_DKL.N


    # LEVEL OF FIDELITY: 0
    model_0 = MF_DKL.add_level(level=0, latent_dim=latent_dim)
    model_0, train_loader_0 = MF_DKL.train_level(level=0, model=model_0)
    z_0, z_next_0, z_fwd_0, _, _ = MF_DKL.eval_level(model_0, train_loader_0)

    z_0 = z_0.detach()
    z_next_0 = z_next_0.detach()
    z_fwd_0 = z_fwd_0.detach()
    ID_0 = estimate_ID(z_0, z_next_0, z_fwd_0)


    # LEVEL OF FIDELITY: 1
    model_1 = MF_DKL.add_level(level=1, latent_dim=latent_dim)
    model_1, train_loader_1 = MF_DKL.train_level(level=1, model=model_1)
    z_1, z_next_1, z_fwd_1, _, _ = MF_DKL.eval_level(model_1, train_loader_1)

    z_1 = z_1.detach()
    z_next_1 = z_next_1.detach()
    z_fwd_1 = z_fwd_1.detach()
    ID_1 = estimate_ID(z_1, z_next_1, z_fwd_1)


    # LEVEL OF FIDELITY: 2
    model_2 = MF_DKL.add_level(level=2, latent_dim=latent_dim)
    model_2, train_loader_2 = MF_DKL.train_level(level=2, model=model_2)
    z_2, z_next_2, z_fwd_2, _, _ = MF_DKL.eval_level(model_2, train_loader_2)

    z_2 = z_2.detach()
    z_next_2 = z_next_2.detach()
    z_fwd_2 = z_fwd_2.detach()
    ID_2 = estimate_ID(z_2, z_next_2, z_fwd_2)


    # Compute a summary of the latent representation and a final estimate of ID
    z_LF = z_0[0 : N[3]] + z_1[0 : N[3]] + z_2[0 : N[3]]
    z_next_LF = z_next_0[0 : N[3]] + z_next_1[0 : N[3]] + z_next_2[0 : N[3]]
    z_fwd_LF = z_fwd_0[0 : N[3]] + z_fwd_1[0 : N[3]] + z_fwd_2[0 : N[3]]

    ID = int((ID_0 + ID_1 + ID_2) / 3) 
    print("ID = ", ID)

    # LEVEL OF FIDELITY: 3
    model_3 = MF_DKL.add_level(level=3, latent_dim=ID, latent_dim_LF=latent_dim)
    model_3 = MF_DKL.train_level(level=3, model=model_3, z_LF=z_LF, z_next_LF=z_next_LF, z_fwd_LF=z_fwd_LF)

    print("##############################################")
    print("ID = ", ID)

    # Save the estimate of ID to file (same folder as the weights)
    f = open(MF_DKL.save_pth_dir + "/ID.txt", "a")
    f.write("ID = " + str(ID))
    f.close()

if __name__ == "__main__":
    main()
