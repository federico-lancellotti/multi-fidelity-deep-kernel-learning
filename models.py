from typing import Union
from linear_operator.operators import LinearOperator
import torch
from torch import Tensor
from torch.distributions import Distribution
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import math
import yaml
import numpy as np

from intrinsic_dimension import eval_id

with open("config.yaml", "r") as file:
    args = yaml.safe_load(file)
OUT_DIM = args["out_dim"]

################################
##### Reconstruction model #####
################################

# The encoder is composed of 4 convolutional layers with 32 filters per layer.
# The convolutional filters are of size (3 × 3) and shifted across the images
# with stride 1 (only the first convolutional layer has stride 2 to quickly
# reduce the input dimensionality). Batch normalization is also used after
# the 2nd and 4th convolutional layers.
# The output features of the last convolutional layer are flattened and fed to
# two final fully connected layers of dimensions 256 and 20, respectively,
# compressing the features to a 20-dimensional feature vector. Each layer has ELU
# activations, except the last fully-connected layer with a linear activation.
class Encoder(nn.Module):
    def __init__(self, hidden_dim=256, z_dim=20, out_dim=84):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=(3, 3), stride=(2, 2)
        )  # stride 2 to quickly reduce the input dimensionality
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(
            32, 32, kernel_size=(3, 3), stride=(1, 1)
        )  # (first) input shape (84,84,6) -> output shape (35,35,32)
        self.batch2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * out_dim * out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)

    def encoder(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.batch1(x)
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.batch2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        return self.encoder(x)


# The decoder is composed of a linear fully-connected layer and 4 transpose
# convolutional layers with 32 filters each. The convolutional filters are of
# size (3 × 3) and shifted across the images with stride 1 (again, the last
# convolutional layer has stride 2). Batch normalization is used after the
# 2nd and 4th convolutional layers, and ELU activations are employed for all
# the layers except the last one.
# The outputs are the mean μ_xhat_t and variance sigma2_xhat_t of N(mu_xhat_t, sigma2_xhat_t).
class Decoder(nn.Module):
    def __init__(self, z_dim=20, out_dim=84):
        super(Decoder, self).__init__()

        # decoder part
        self.fc = nn.Linear(z_dim, 32 * out_dim * out_dim)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, out_dim, out_dim))
        self.batch3 = nn.BatchNorm2d(32)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.batch4 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.deconv4 = nn.ConvTranspose2d(
            32, 6, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1)
        )

    def decoder(self, z):
        z = F.elu(self.fc(z))
        z = self.unflatten(z)
        z = self.batch3(z)
        z = F.elu(self.deconv1(z))
        z = F.elu(self.deconv2(z))
        z = self.batch4(z)
        z = F.elu(self.deconv3(z))
        mu = F.sigmoid(self.deconv4(z))
        std = torch.ones_like(mu).detach()
        return mu, std

    def forward(self, z):
        return self.decoder(z)


# The latent variables of the feature vector are fed to independent GPs with
# constant mean and ARD-SE kernel, which produces a 20-dimensional latent state
# distribution p( z_t | x_t ). From the latent state distribution p( z_t | x_t ), we
# can sample the latent state vectors z_t.
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_size=64, grid_bounds=(-10.0, 10)):
        # num_dim: number of dimensions (tasks) in the GP model, namely the dimensionality of the output space.
        # grid_size: the grid is a set of points used to approximate the GP model. A larger grid size provides a denser approximation.
        # grid_bounds: specifies the bounds of the regularization.

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.GridInterpolationVariationalStrategy(
                    self,
                    grid_size=grid_size,
                    grid_bounds=[grid_bounds],
                    variational_distribution=variational_distribution,
                ),
                num_tasks=num_dim,
            )
        )
        super().__init__(
            variational_strategy
        )  # calls the constructor of the base class (gpytorch.models.ApproximateGP) and initializes the GP model with the specified variational strategy

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=gpytorch.kernels.RBFKernel(
                ard_num_dims=num_dim,
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                ),
            )
        )
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# Complete autoencoder.
# The latent state vectors z_t are fed into the decoder D to learn
# the reconstruction distribution p(xˆt|zt). A popular choice for p(x_t|z_t)
# is Gaussian with unit variance.
class SVDKL_AE(gpytorch.Module):
    def __init__(
        self,
        num_dim,
        likelihood,
        grid_bounds=(-10.0, 10.0),
        hidden_dim=32,
        grid_size=32,
        obs_dim=84,
        rho=1,
    ):
        super(SVDKL_AE, self).__init__()
        self.num_dim = num_dim
        self.out_dim = OUT_DIM[obs_dim]
        self.rho = rho
        self.grid_bounds = grid_bounds
        self.likelihood = likelihood

        self.gp_layer = GaussianProcessLayer(num_dim, grid_size, grid_bounds)
        self.encoder = Encoder(hidden_dim, self.num_dim, self.out_dim)
        self.decoder = Decoder(self.num_dim, self.out_dim)

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            self.grid_bounds[0], self.grid_bounds[1]
        )

    def forward(self, x, z_LF):
        features = self.encoder(x) + self.rho*z_LF
        features = self.scale_to_bounds(features)
        features = features.transpose(-1, -2).unsqueeze(
            -1
        )  # This line makes it so that we learn a GP for each feature

        if self.training:
            with gpytorch.settings.detach_test_caches(False):
                self.gp_layer.train()
                self.gp_layer.eval()
                res = self.gp_layer(features)
        else:
            res = self.gp_layer(features)

        mean = res.mean
        covar = res.variance
        z = self.likelihood(res).rsample()

        mu_hat, var_hat = self.decoder.decoder(z)

        return mu_hat, var_hat, res, mean, covar, z


# Two step autoencoder.
# The (external) latent representation is used as input of an internal
# autoencoder with latent space dimension equal to ID.
class SVDKL_AE_2step(gpytorch.Module):
    def __init__(
        self,
        num_dim,
        likelihood,
        grid_bounds=(-10.0, 10.0),
        hidden_dim=32,
        grid_size=32,
        obs_dim=84,
        rho=1,
        ID=0,
    ):
        super(SVDKL_AE_2step, self).__init__()
        self.num_dim = num_dim
        self.out_dim = OUT_DIM[obs_dim]
        self.rho = rho
        self.grid_bounds = grid_bounds
        self.likelihood = likelihood
        self.ID = ID

        self.gp_layer = GaussianProcessLayer(num_dim, grid_size, grid_bounds)
        self.ext_encoder = Encoder(hidden_dim, self.num_dim, self.out_dim)
        self.ext_decoder = Decoder(self.num_dim, self.out_dim)
        
        if self.ID:
            self.int_encoder = nn.Sequential(
                nn.Linear(self.num_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.ID)
            )
            self.int_decoder = nn.Sequential(
                nn.Linear(self.ID, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_dim)
            )
        else:
            self.int_encoder = None
            self.int_decoder = None

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            self.grid_bounds[0], self.grid_bounds[1]
        )

    def forward(self, x, z_LF):
        features = self.ext_encoder(x)
        features = self.scale_to_bounds(features)
        features = features.transpose(-1, -2).unsqueeze(
            -1
        )  # This line makes it so that we learn a GP for each feature

        if self.training:
            with gpytorch.settings.detach_test_caches(False):
                self.gp_layer.train()
                self.gp_layer.eval()
                res = self.gp_layer(features)
        else:
            res = self.gp_layer(features)

        mean = res.mean
        covar = res.variance
        z_LF = torch.tensor(
            z_LF.astype(np.float32)
        )  # trasform the numpy array for compatibility
        z = self.likelihood(res).rsample() + self.rho*z_LF

        # Use the internal autoencoder only if ID is properly defined
        if self.ID:
            y = self.int_encoder(z)
            z = self.int_decoder(y)
        else:
            y = z 

        mu_hat, var_hat = self.ext_decoder.decoder(z)

        return mu_hat, var_hat, res, mean, covar, y


################################
##### Reconstruction model #####
################################
    
class SVDKL_AE_latent_dyn(nn.Module):
    def __init__(self, num_dim, likelihood, likelihood_fwd, grid_bounds=(-10., 10.), h_dim=32, grid_size=32, obs_dim=84, rho=1):
        super(SVDKL_AE_latent_dyn, self).__init__()

        self.obs_dim = obs_dim

        self.AE_DKL = SVDKL_AE(num_dim=num_dim, likelihood=likelihood, grid_bounds=grid_bounds, hidden_dim=h_dim, 
                               grid_size=grid_size, obs_dim=obs_dim, rho=rho)
        self.fwd_model_DKL = Forward_DKLModel(num_dim=num_dim, grid_bounds=grid_bounds, h_dim=h_dim,
                                              grid_size=grid_size, likelihood=likelihood_fwd, rho=rho)  # DKL forward model

    def forward(self, x, z_LF, x_next, z_next_LF, z_fwd_LF):
        mu_x, var_x, res, mu, var, z = self.AE_DKL(x, z_LF)
        mu_x_target, var_x_target, res_target, mu_target, var_target, z_target = self.AE_DKL(x_next, z_next_LF)
        res_fwd, mu_fwd, var_fwd, z_fwd = self.fwd_model_DKL(z, z_fwd_LF)
        return mu_x, var_x, mu, var, z, res, mu_target, var_target, z_target, res_target, mu_fwd, var_fwd, res_fwd, z_fwd

    def predict_dynamics(self, z, z_fwd_LF, samples=1):
        res_fwd, mu_fwd, var_fwd, z_fwd = self.fwd_model_DKL(z, z_fwd_LF)
        if samples == 1:
            mu_x_rec, _ = self.AE_DKL.decoder(z_fwd)
        else:
            mu_x_recs = torch.zeros((samples, 6, self.obs_dim, self.obs_dim))
            z_fwd = self.fwd_model_DKL.likelihood(res_fwd).sample(sample_shape=torch.Size([samples]))
            for i in range(z_fwd.shape[0]):
                mu_x_recs[i], _ = self.AE_DKL.decoder(z_fwd[i])
            mu_x_rec = mu_x_recs.mean(0)
            z_fwd = z_fwd.mean(0)
        return mu_x_rec, z_fwd, mu_fwd, res_fwd

    def predict_dynamics_mean(self, mu, z_fwd_LF):
        res_fwd, mu_fwd, var_fwd, z_fwd = self.fwd_model_DKL(mu, z_fwd_LF)
        mu_x_rec, _ = self.AE_DKL.decoder(mu_fwd)
        return mu_x_rec, z_fwd, mu_fwd, res_fwd
    

class Forward_DKLModel(gpytorch.Module):
    def __init__(self, num_dim, likelihood, grid_bounds=(-10., 10.), h_dim=256, grid_size=32, rho=1):
        super(Forward_DKLModel, self).__init__()
        self.gp_layer_2 = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds, grid_size=grid_size)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim
        self.likelihood = likelihood
        self.rho = rho

        self.fwd_model = ForwardModel(num_dim, h_dim) # NN model

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x, z_LF):
        features = self.fwd_model(x) + self.rho*z_LF
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        if self.training:
            with gpytorch.settings.detach_test_caches(False):
                self.gp_layer_2.train()
                self.gp_layer_2.eval()
                res = self.gp_layer_2(features)
        else:
            res = self.gp_layer_2(features)
        mean = res.mean
        var = res.variance
        z = self.likelihood(res).rsample()
        return res, mean, var, z
    

class ForwardModel(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(ForwardModel, self).__init__()

        self.fc = nn.Linear(z_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.batch = nn.BatchNorm1d(z_dim)

    def forward(self, z):
        z = F.elu(self.fc(z))
        z = F.elu(self.fc1(z))
        features = self.fc2(z)
        return features