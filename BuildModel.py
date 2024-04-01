import torch
import numpy as np
import os
import gpytorch
import gc
from datetime import datetime
import yaml

from models import SVDKL_AE, SVDKL_AE_2step
from models import SVDKL_AE_latent_dyn
from variational_inference import VariationalKL
from logger import Logger
from utils import load_pickle, plot_frame
from trainer import train_dyn as train
from data_loader import DataLoader
from intrinsic_dimension import eval_id

class BuildModel:
    def __init__(self, args, use_gpu=False):
        self.use_gpu = use_gpu
        
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
        self.exp = args["exp"]
        self.mtype = args["mtype"]
        self.training_dataset = args["training_dataset"]
        self.log_interval = args["log_interval"]
        self.jitter = float(args["jitter"])

        # Load data
        levels = len(self.training_dataset)
        self.directory = os.path.dirname(os.path.abspath(__file__))

        self.folder = []
        for l in range(levels):
            self.folder.append(os.path.join(self.directory + "/Data", self.training_dataset[l]))

        self.data = []
        self.N = []
        for l in range(levels):
            self.data.append(load_pickle(self.folder[l]))
            self.N.append(len(self.data[l]))

        # Set the string with date-time for the saving folder
        now = datetime.now()
        self.folder_date = now.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")

        # Set the folder for saving the results
        self.save_pth_dir = self.directory + "/Results/" + str(self.exp) + "/" + str(self.mtype) + "/" + self.folder_date + "/"
        if not os.path.exists(self.save_pth_dir):
            os.makedirs(self.save_pth_dir)


    def add_level(self, level, latent_dim, z_LF=None, z_next_LF=None, z_fwd_LF=None, latent_dim_LF=0):
        # Define dummy previous level of fidelity
        if z_LF == None or z_next_LF == None or z_fwd_LF == None or latent_dim_LF == 0:
            z_LF = torch.zeros((self.N[level], latent_dim))
            z_next_LF = torch.zeros((self.N[level], latent_dim))
            z_fwd_LF = torch.zeros((self.N[level], latent_dim))
            latent_dim_LF = latent_dim

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

        if self.use_gpu:
            if torch.cuda.is_available():
                model = model.cuda()
                gc.collect()  # NOTE: Critical to avoid GPU leak
            elif torch.backends.mps.is_available():  # on Apple Silicon
                mps_device = torch.device("mps")
                model.to(mps_device)

        variational_kl_term = VariationalKL(
            model.AE_DKL.likelihood, model.AE_DKL.gp_layer, num_data=self.batch_size
        )  # len(data)
        variational_kl_term_fwd = VariationalKL(
            model.fwd_model_DKL.likelihood,
            model.fwd_model_DKL.gp_layer_2,
            num_data=self.batch_size,
        )

        # Use the adam optimizer
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

        optimizer_var1 = torch.optim.SGD([
            {'params': model.AE_DKL.gp_layer.variational_parameters(), 'lr': self.lr_gp_var},
            ], lr=self.lr_gp_var, momentum=0.9, nesterov=True, weight_decay=0) # momentum=0.9, nesterov=True, weight_decay=0
        optimizer_var2 = torch.optim.SGD([
            {'params': model.fwd_model_DKL.gp_layer_2.variational_parameters(), 'lr': self.lr_gp_var},
            ], lr=self.lr_gp_var, momentum=0.9, nesterov=True, weight_decay=0)

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

        # Preprocessing of the data
        train_loader = DataLoader(
            self.data[level],
            z_LF,
            z_next_LF,
            z_fwd_LF,
            obs_dim=(self.obs_dim_1[level], self.obs_dim_2[level], self.obs_dim_3),
        )

        # Training
        if self.training:
            print("start training...")
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


    def eval_level(self, model, train_loader):
        model.eval()
        model.AE_DKL.likelihood.eval()
        model.fwd_model_DKL.likelihood.eval()    

        pred_samples = train_loader.get_all_samples()
        pred_samples["obs"] = pred_samples["obs"].permute(0, 3, 1, 2)
        pred_samples["next_obs"] = pred_samples["next_obs"].permute(0, 3, 1, 2)

        _, _, _, _, z, _, _, _, z_next, _, _, _, _, z_fwd = model(
            pred_samples["obs"],
            pred_samples["z_LF"],
            pred_samples["next_obs"],
            pred_samples["z_next_LF"],
            pred_samples["z_fwd_LF"],
        )

        return z, z_next, z_fwd