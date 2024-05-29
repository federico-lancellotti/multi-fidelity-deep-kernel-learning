import numpy as np
import torch
import os
import gpytorch
import gc
from datetime import datetime

from models import SVDKL_AE_latent_dyn
from variational_inference import VariationalKL
from utils import load_pickle
from trainer import train
from data_loader import BaseDataLoader, GymDataLoader, PDEDataLoader

import warnings
warnings.filterwarnings("ignore", message="torch.sparse.SparseTensor")


class BuildModel:
    """
    Class for building and training a multi-fidelity deep kernel learning model.

    Args:
        args (dict): Dictionary containing the model parameters.
        use_gpu (bool, optional): Flag indicating whether to use GPU for training. Defaults to False.
        test (bool, optional): Flag indicating whether the model is being used for testing. Defaults to False.

    Attributes:
        use_gpu (bool): Flag indicating whether to use GPU for training.
        seed (int): The seed for the random number generator.
        batch_size (int): The size of the batch.
        max_epoch (int): The maximum number of epochs for training.
        rho (float): The rho parameter for the model.
        training (bool): Flag indicating whether the model is being trained.
        lr (float): The learning rate for the model.
        lr_gp (float): The learning rate for the GP layer.
        lr_gp_lik (float): The learning rate for the likelihood.
        lr_gp_var (float): The learning rate for the variational parameters.
        reg_coef (float): The regularization coefficient.
        k1 (float): The k1 parameter for the model.
        k2 (float): The k2 parameter for the model.
        grid_size (int): The size of the grid.
        obs_dim_1 (list): The first dimension of the observation, at each fidelity level.
        obs_dim_2 (list): The second dimension of the observation, at each fidelity level.
        obs_dim_3 (int): The third dimension of the observation.
        h_dim (int): The dimension of the hidden layer.
        env_name (str): The name of the environment.
        training_dataset (list): The list of training datasets.
        log_interval (int): The interval for logging.
        testing_dataset (list): The list of testing datasets.
        jitter (float): The jitter parameter for the model, to avoid numerical instability.
        results_folder (str): The folder for saving the results.
        weights_filename (list): The list of filenames for saving the weights.
        folder (list): The list of folders containing the data.
        data (list): The list of data samples.
        N (list): The list of the number of data samples.
        directory (str): The directory of the model.
        folder_date (str): The folder for saving the results.
        save_pth_dir (str): The path for saving the results.

    Methods:
        add_level(level, latent_dim, latent_dim_LF=0): Adds a level of fidelity to the model.
        train_level(level, model, z_LF=None, z_next_LF=None, z_fwd_LF=None): Trains a level of fidelity in the model.
        test_level(level, model, z_LF=None, z_next_LF=None, z_fwd_LF=None): Tests a level of fidelity in the model.
        eval_level(model, data_loader): Evaluates a level of fidelity in the model.
    """

    def __init__(self, args, use_gpu=False, test=False):
        """
        Initialize the BuildModel object.
        Set the parameters for the model and load the data.

        Args:
            args (dict): A dictionary containing the parameters for the model.
            use_gpu (bool, optional): Whether to use GPU for computations. Defaults to False.
            test (bool, optional): Whether the model is being used for testing. Defaults to False.
        """

        self.test = test

        self.use_gpu = use_gpu
        self.device = torch.device("cpu")
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
        
        # Set parameters
        self.seed = args["seed"]
        self.batch_size = args["batch_size"]
        self.max_epoch = args["max_epoch"]
        self.rho = args["rho"]
        self.training = args["training"]
        self.lr = float(args["lr"])
        self.lr_gp = float(args["lr_gp"])
        self.lr_gp_lik = float(args["lr_gp_lik"])
        self.lr_gp_var = float(args["lr_gp_var"])
        self.reg_coef = float(args["reg_coef"])
        self.k1 = float(args["k1"])
        self.k2 = float(args["k2"])
        self.grid_size = args["grid_size"]
        self.obs_dim_1 = args["obs_dim_1"]
        self.obs_dim_2 = args["obs_dim_2"]
        self.obs_dim_3 = args["obs_dim_3"]
        self.h_dim = args["h_dim"]
        self.env_name = args["env_name"]
        self.training_dataset = args["training_dataset"]
        self.log_interval = args["log_interval"]
        self.jitter = float(args["jitter"])

        self.testing_dataset = args["testing_dataset"]
        self.results_folder = args["results_folder"]
        self.weights_filename = args["weights_filename"]

        # Load data
        levels = len(self.training_dataset)
        self.directory = os.path.dirname(os.path.abspath(__file__))

        self.folder = []
        self.data = []
        self.N = []

        if self.test: 
            for l in range(levels):
                self.folder.append(os.path.join(self.directory + "/Data", self.testing_dataset[l]))
        else:
            for l in range(levels):
                self.folder.append(os.path.join(self.directory + "/Data", self.training_dataset[l]))
        
        for l in range(levels):
            self.data.append(load_pickle(self.folder[l]))
            self.N.append(len(self.data[l]))

        if not self.test:
            # Set the string with date-time for the saving folder
            now = datetime.now()
            self.folder_date = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")

            # Set the folder for saving the results
            self.save_pth_dir = self.directory + "/Results/" + self.env_name + "/" + self.folder_date + "/"
            if not os.path.exists(self.save_pth_dir):
                os.makedirs(self.save_pth_dir)

        # Select the correct data loader with respect to the environment
        case_to_loader = {
            "Pendulum": GymDataLoader,
            "Acrobot": GymDataLoader,
            "MountainCarContinuous": GymDataLoader,
            "reaction-diffusion": PDEDataLoader,
        }
        self.data_loader = case_to_loader.get(self.env_name, BaseDataLoader)


    def add_level(self, level, latent_dim, latent_dim_LF=0):
        """
        Add a new instance of the model with the specified level of fidelity.
        Set the likelihood and initialize the model.

        Args:
            level (int): The level to be added.
            latent_dim (int): The dimension of the latent space.
            latent_dim_LF (int, optional): The dimension of the low-fidelity latent space. Defaults to 0.

        Returns:
            model: The initialized model with the added level.
        """

        # Set likelihood
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            latent_dim, rank=0, has_task_noise=True, has_global_noise=False
        )
        likelihood_fwd = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            latent_dim, rank=0, has_task_noise=True, has_global_noise=False
        )

        # Model initialization
        model = SVDKL_AE_latent_dyn(
            num_dim=latent_dim,
            likelihood=likelihood,
            likelihood_fwd=likelihood_fwd,
            grid_bounds=(-10.0, 10.0),
            h_dim=self.h_dim,
            grid_size=self.grid_size,
            obs_dim=self.obs_dim_1[level],
            rho=self.rho,
            num_dim_LF=latent_dim_LF,
        )

        return model


    def train_level(self, level, model, z_LF=None, z_next_LF=None, z_fwd_LF=None):
        """
        Trains the model instance at a specific level of fidelity.

        Args:
            level (int): The level of fidelity.
            model: The model to be trained.
            z_LF (torch.Tensor, optional): The previous level of fidelity for the low-fidelity data. Defaults to None.
            z_next_LF (torch.Tensor, optional): The next level of fidelity for the low-fidelity data. Defaults to None.
            z_fwd_LF (torch.Tensor, optional): The forward level of fidelity for the low-fidelity data. Defaults to None.

        Returns:
            model: The trained model.
            train_loader: The data loader used for training.
        """

        latent_dim = model.num_dim
        
        # Define dummy previous level of fidelity
        if z_LF == None or z_next_LF == None or z_fwd_LF == None:
            z_LF = torch.zeros((self.N[level], latent_dim))
            z_next_LF = torch.zeros((self.N[level], latent_dim))
            z_fwd_LF = torch.zeros((self.N[level], latent_dim))

        if self.use_gpu:
            model = model.to(self.device)
            if torch.cuda.is_available():
                gc.collect()  # NOTE: Critical to avoid GPU leak

        # Define the variational loss
        variational_kl_term = VariationalKL(
            model.AE_DKL.likelihood, 
            model.AE_DKL.gp_layer, 
            num_data=self.batch_size
        )
        variational_kl_term_fwd = VariationalKL(
            model.fwd_model_DKL.likelihood,
            model.fwd_model_DKL.gp_layer_2,
            num_data=self.batch_size,
        )

        # Set the optimizer
        ## Set the Adam optimizer
        optimizer = torch.optim.Adam(
            [
                {"params": model.AE_DKL.encoder.parameters()},
                {"params": model.AE_DKL.decoder.parameters()},
                {"params": model.fwd_model_DKL.fwd_model.parameters()},
                {"params": model.AE_DKL.gp_layer.hyperparameters(), "lr": self.lr_gp},
                {"params": model.fwd_model_DKL.gp_layer_2.hyperparameters(), "lr": self.lr_gp},
                {"params": model.AE_DKL.likelihood.parameters(), "lr": self.lr_gp_lik},
                {"params": model.fwd_model_DKL.likelihood.parameters(), "lr": self.lr_gp_lik},
            ],
            lr=self.lr,
            weight_decay=self.reg_coef,
        )

        ## Use the SGD optimizer for the variational parameters
        optimizer_var1 = torch.optim.SGD([
            {'params': model.AE_DKL.gp_layer.variational_parameters(), 'lr': self.lr_gp_var},
            ], lr=self.lr_gp_var, momentum=0.9, nesterov=True, weight_decay=0) # momentum=0.9, nesterov=True, weight_decay=0
        optimizer_var2 = torch.optim.SGD([
            {'params': model.fwd_model_DKL.gp_layer_2.variational_parameters(), 'lr': self.lr_gp_var},
            ], lr=self.lr_gp_var, momentum=0.9, nesterov=True, weight_decay=0)

        ## Collect the optimizers in a list
        optimizers = [optimizer, optimizer_var1, optimizer_var2]

        # Reduce the learning rate when at 50% and 75% of the training.
        scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_var1, milestones=[0.5 * self.max_epoch, 0.75 * self.max_epoch], gamma=0.1
        )
        scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_var2, milestones=[0.5 * self.max_epoch, 0.75 * self.max_epoch], gamma=0.1
        )

        # Set how to save the model
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")

        # Define the data loader for training
        
        train_loader = self.data_loader(
            self.data[level],
            z_LF,
            z_next_LF,
            z_fwd_LF,
            obs_dim=(self.obs_dim_1[level], self.obs_dim_2[level], self.obs_dim_3),
        )

        # Train the model
        if self.training:
            print("Start training...", flush=True)
            for epoch in range(1, self.max_epoch + 1):
                with gpytorch.settings.cholesky_jitter(self.jitter):
                    train(
                        model=model,
                        train_data=train_loader,
                        optimizers=optimizers,
                        batch_size=self.batch_size,
                        num_epochs=epoch,
                        variational_kl_term=variational_kl_term,
                        variational_kl_term_fwd=variational_kl_term_fwd,
                        k1=self.k1,
                        beta=self.k2,
                        use_gpu=self.use_gpu,
                    )

                    scheduler_1.step()
                    scheduler_2.step()

                if epoch % self.log_interval == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "likelihood": model.AE_DKL.likelihood.state_dict(),
                            "likelihood_fwd": model.fwd_model_DKL.likelihood.state_dict(),
                        },
                        self.save_pth_dir
                        + "/MFDKL_dyn_level"
                        + str(level)
                        + "_"
                        + str(self.obs_dim_1[level])
                        + "_"
                        + date_string
                        + ".pth",
                    )

            torch.save(
                {
                    "model": model.state_dict(),
                    "likelihood": model.AE_DKL.likelihood.state_dict(),
                    "likelihood_fwd": model.fwd_model_DKL.likelihood.state_dict(),
                },
                self.save_pth_dir
                + "/MFDKL_dyn_level"
                + str(level)
                + "_"
                + str(self.obs_dim_1[level])
                + "_"
                + date_string
                + ".pth",
            )

        return model, train_loader


    def test_level(self, level, model, z_LF=None, z_next_LF=None, z_fwd_LF=None):
        """
        Test the model at a specific fidelity level.

        Args:
            level (int): The fidelity level to test the model at.
            model: The model to be tested.
            z_LF (torch.Tensor, optional): The latent variable at the previous level of fidelity. Defaults to None.
            z_next_LF (torch.Tensor, optional): The latent variable at the next level of fidelity. Defaults to None.
            z_fwd_LF (torch.Tensor, optional): The latent variable at the forward level of fidelity. Defaults to None.

        Returns:
            tuple: A tuple containing the model and the data loader.
        """

        # Load data
        directory = os.path.dirname(os.path.abspath(__file__))
        data = self.data[level]
        N = self.N[level]

        # Load weights
        weights_folder = os.path.join(
            directory + "/Results/" + self.env_name + "/" + self.results_folder + "/", self.weights_filename[level] + ".pth"
        )
        
        model.load_state_dict(torch.load(weights_folder)["model"])
        model.AE_DKL.likelihood.load_state_dict(torch.load(weights_folder)["likelihood"])
        model.fwd_model_DKL.likelihood.load_state_dict(torch.load(weights_folder)["likelihood_fwd"])

        # Define dummy previous level of fidelity
        if z_LF == None or z_next_LF == None or z_fwd_LF == None:
            latent_dim = model.num_dim
            z_LF = torch.zeros((N, latent_dim))
            z_next_LF = torch.zeros((N, latent_dim))
            z_fwd_LF = torch.zeros((N, latent_dim))

        # Define the data loader for testing
        data_loader = DataLoader(
            data, z_LF, z_next_LF, z_fwd_LF, obs_dim=(self.obs_dim_1[level], self.obs_dim_2[level], self.obs_dim_3)
        )

        return model, data_loader


    def get_latent_representations(self, model, data_loader, idx=range(100), step=600):
        """
        Get the latent representations of the samples in the data loader.

        Args:
            model (Model): The model to use for the latent representation.
            data_loader (DataLoader): The data loader containing the samples.
            idx (list, optional): The indices of the samples to get the latent representation of. Defaults to range(100).
            step (int, optional): The number of samples to evaluate at a time. Defaults to 600.

        Returns:
            tuple: A tuple containing: 
                    - the latent representations of the samples at time t-1 and t, 
                    - the next latent representations at time t and t+1, and 
                    - the forward predicted latent representations at time t and t+1.
        """

        # Set the model to evaluation mode
        model.eval()
        model.AE_DKL.likelihood.eval()
        model.fwd_model_DKL.likelihood.eval()

        # Clear the cache
        torch.cuda.empty_cache()
        gc.collect()

        # Set the lists to store the latent representations
        z_list = []
        z_next_list = []
        z_fwd_list = []

        # Split the indices into chunks
        chunks = [idx[i:min(i+step,len(idx))] for i in range(0, len(idx), step)]
        
        for idx_chunk in chunks:
            # Sample the batch with the given set of indices
            pred_samples = data_loader.sample_batch_with_idx(idx_chunk)

            # Prepare the data for the model evaluation
            pred_samples["obs"] = pred_samples["obs"].permute(0, 3, 1, 2).to(self.device)
            pred_samples["next_obs"] = pred_samples["next_obs"].permute(0, 3, 1, 2).to(self.device)
            pred_samples["z_LF"] = pred_samples["z_LF"].to(self.device)
            pred_samples["z_next_LF"] = pred_samples["z_next_LF"].to(self.device)
            pred_samples["z_fwd_LF"] = pred_samples["z_fwd_LF"].to(self.device)

            # Evaluate the model on the current batch
            _, _, _, z, _, _, _, z_next, _, _, _, _, _, z_fwd, _ = model(
                pred_samples["obs"],
                pred_samples["z_LF"],
                pred_samples["next_obs"],
                pred_samples["z_next_LF"],
                pred_samples["z_fwd_LF"],
            )

            # Store the new latent representations
            z_list.append(z.detach())
            z_next_list.append(z_next.detach())
            z_fwd_list.append(z_fwd.detach())

        # Concatenate the latent representations
        z = torch.cat(z_list, dim=0)
        z_next = torch.cat(z_next_list, dim=0)
        z_fwd = torch.cat(z_fwd_list, dim=0)

        return z, z_next, z_fwd


    def eval_level(self, model, data_loader):
        """
        Evaluate the model at a given level.

        Args:
            model (Model): The model to evaluate.
            data_loader (DataLoader): The data loader containing the samples.

        Returns:
            tuple: A tuple containing the following elements:
                - z (Tensor): The latent representation of the sample frames at time t-1 and t.
                - z_next (Tensor): The latent representation of the sample frames at time t and t+1.
                - z_fwd (Tensor): The latent representation of the forward predicted samples, at time t and t+1.
                - mu_x (Tensor): The mean of the observed samples, at time t-1 and t.
                - mu_next (Tensor): The mean of the next observed samples, at time t and t+1.
        """

        # Set the model to evaluation mode
        model.eval()
        model.AE_DKL.likelihood.eval()
        model.fwd_model_DKL.likelihood.eval()

        # Clear the cache
        torch.cuda.empty_cache()
        gc.collect()

        # Get all the samples
        pred_samples = data_loader.get_all_samples()
        pred_samples["obs"] = pred_samples["obs"].permute(0, 3, 1, 2).to(self.device)
        pred_samples["next_obs"] = pred_samples["next_obs"].permute(0, 3, 1, 2).to(self.device)
        pred_samples["z_LF"] = pred_samples["z_LF"].to(self.device)
        pred_samples["z_next_LF"] = pred_samples["z_next_LF"].to(self.device)
        pred_samples["z_fwd_LF"] = pred_samples["z_fwd_LF"].to(self.device)

        # Evaluate the model on the current batch
        mu_x, _, _, _, z, _, mu_next, _, z_next, _, _, _, _, z_fwd, _ = model(
            pred_samples["obs"],
            pred_samples["z_LF"],
            pred_samples["next_obs"],
            pred_samples["z_next_LF"],
            pred_samples["z_fwd_LF"],
        )

        # Detach the tensors
        z = z.detach()
        z_next = z_next.detach()
        z_fwd = z_fwd.detach()
        mu_x = mu_x.detach()
        mu_next = mu_next.detach()

        return z, z_next, z_fwd, mu_x, mu_next