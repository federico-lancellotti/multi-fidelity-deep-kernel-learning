import torch
import numpy as np
import yaml

from src.intrinsic_dimension import estimate_ID
from src.BuildModel import BuildModel
from src.utils import align_pde, get_length


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

    # Compute indices to align the levels of fidelity in time
    T = args['T']
    dt = args['dt']
    mu = args['mu']

    idx = align_pde(len_0=int(T[0]/dt), len_1=int(T[1]/dt), n_cases_0=get_length(mu[0]), n_cases_1=get_length(mu[1]))


    # LEVEL OF FIDELITY: 0
    model_0 = MF_DKL.add_level(level=0, latent_dim=latent_dim)
    model_0, train_loader_0 = MF_DKL.train_level(level=0, model=model_0)
    z_0, z_next_0, z_fwd_0 = MF_DKL.get_latent_representations(model_0, train_loader_0, idx)

    ID = estimate_ID(z_0, z_next_0, z_fwd_0)


    # Compute a summary of the latent representation and a final estimate of ID
    z_LF = z_0
    z_next_LF = z_next_0
    z_fwd_LF = z_fwd_0

    print("ID = ", ID)

    # LEVEL OF FIDELITY: 1
    model_1 = MF_DKL.add_level(level=1, latent_dim=ID, latent_dim_LF=latent_dim)
    model_1 = MF_DKL.train_level(level=1, model=model_1, z_LF=z_LF, z_next_LF=z_next_LF, z_fwd_LF=z_fwd_LF)


    print("##############################################")
    print("ID = ", ID)

    # Save the estimate of ID to file (same folder as the weights)
    f = open(MF_DKL.save_pth_dir + "/ID.txt", "a")
    f.write("ID = " + str(ID))
    f.close()

if __name__ == "__main__":
    main()
