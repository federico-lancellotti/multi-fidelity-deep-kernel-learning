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
#obs_dim = args["obs_dim_1"]
#out_dim = [OUT_DIM[obs_dim[0]], OUT_DIM[obs_dim[1]]]


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
        self.deconv4 = nn.ConvTranspose2d(32, 6, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1))

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
        features = self.encoder(x)
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
        z_LF = torch.tensor(z_LF.astype(np.float32))    # trasform the numpy array for compatibility
        z = self.likelihood(res).rsample() + z_LF 

        mu_hat, var_hat = self.decoder.decoder(z)

        return mu_hat, var_hat, res, mean, covar, z


# # Multi fidelity version of SVDKL_AE
# class MF_SVDKL_AE(gpytorch.Module):
#     def __init__(
#         self,
#         num_dim,
#         likelihood,
#         grid_bounds=(-10.0, 10.0),
#         hidden_dim=32,
#         grid_size=32,
#         rho=1,
#     ):
#         super(MF_SVDKL_AE, self).__init__()
#         self.num_dim = num_dim
#         self.grid_bounds = grid_bounds
#         self.likelihood = likelihood
#         self.rho = rho

#         self.gp_layer_LF = GaussianProcessLayer(num_dim, grid_size, grid_bounds)
#         self.encoder_LF = Encoder(hidden_dim, self.num_dim, out_dim[0])
#         self.decoder_LF = Decoder(self.num_dim, out_dim[0])

#         self.gp_layer_HF = GaussianProcessLayer(num_dim, grid_size, grid_bounds)
#         self.encoder_HF = Encoder(hidden_dim, self.num_dim, out_dim[1])
#         self.decoder_HF = Decoder(self.num_dim, out_dim[1])

#         # This module will scale the NN features so that they're nice values
#         self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
#             self.grid_bounds[0], self.grid_bounds[1]
#         )

#     def forward(self, x_LF, x_HF):
#         features_LF = self.encoder_LF(x_LF)
#         features_LF = self.scale_to_bounds(features_LF)
#         features_LF = features_LF.transpose(-1, -2).unsqueeze(
#             -1
#         )  # This line makes it so that we learn a GP for each feature

#         if self.training:
#             with gpytorch.settings.detach_test_caches(False):
#                 self.gp_layer_LF.train()
#                 self.gp_layer_LF.eval()
#                 res_LF = self.gp_layer_LF(features_LF)
#         else:
#             res_LF = self.gp_layer_LF(features_LF)

#         z_LF = self.likelihood(res_LF).rsample()

#         mu_hat_LF, var_hat_LF = self.decoder_LF.decoder(z_LF)

#         features_HF = self.encoder_HF(x_HF)
#         features_HF = self.scale_to_bounds(features_HF)
#         features_HF = features_HF.transpose(-1, -2).unsqueeze(
#             -1
#         )  # This line makes it so that we learn a GP for each feature

#         if self.training:
#             with gpytorch.settings.detach_test_caches(False):
#                 self.gp_layer_HF.train()
#                 self.gp_layer_HF.eval()
#                 res_HF = self.gp_layer_HF(features_HF)
#         else:
#             res_HF = self.gp_layer_HF(features_HF)

#         mean_HF = res_HF.mean
#         covar_HF = res_HF.variance
#         z_HF = self.likelihood(res_HF).rsample()

#         z = z_HF + self.rho * z_LF
#         mu_hat_HF, var_hat_HF = self.decoder_HF.decoder(z)

#         return (
#             mu_hat_LF,
#             var_hat_LF,
#             mu_hat_HF,
#             var_hat_HF,
#             res_HF,
#             mean_HF,
#             covar_HF,
#             z,
#         )


# # Reduced one-dimenional encoder for the 2-step model
# class reducedEncoder(nn.Module):
#     def __init__(self, z_dim=5):
#         super(reducedEncoder, self).__init__()

#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
#         self.batch_norm = nn.BatchNorm1d(16)
#         self.linear1 = nn.Linear(256, 16)
#         self.linear2 = nn.Linear(16, z_dim)

#     def encoder(self, x):
#         x = x.unsqueeze(1)
#         x = F.elu(self.conv1(x))
#         x = F.elu(self.conv2(x))
#         x = self.batch_norm(x)
#         x = x.view(x.size(0), -1)
#         x = F.elu(self.linear1(x))
#         x = F.elu(self.linear2(x))
#         return x

#     def forward(self, x):
#         return self.encoder(x)


# # Reduced one-dimensional decoder for the 2-step model
# class reducedDecoder(nn.Module):
#     def __init__(self, z_dim=5, output_dim=20):
#         super(reducedDecoder, self).__init__()

#         self.linear = nn.Linear(z_dim, 256)
#         self.batch_norm = nn.BatchNorm1d(256)
#         self.deconv2 = nn.ConvTranspose1d(in_channels=256, out_channels=32, kernel_size=3)
#         self.deconv1 = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=3, stride=2, output_padding=1)
#         self.linear_output = nn.Linear(8, output_dim)

#     def decoder(self, z):
#         z = F.elu(self.linear(z))
#         z = z.unsqueeze(-1)
#         z = self.batch_norm(z)
#         z = F.elu(self.deconv2(z))
#         z = F.elu(self.deconv1(z))
#         z = z.squeeze(1)
#         z = self.linear_output(z)
#         return z

#     def forward(self, z):
#         return self.decoder(z)


# # 2-step multi fidelity version of SVDKL_AE
# class MF_SVDKL_AE_2step(gpytorch.Module):
    def __init__(
        self,
        num_dim,
        likelihood,
        grid_bounds=(-10.0, 10.0),
        hidden_dim=32,
        grid_size=32,
        rho=1,
    ):
        super(MF_SVDKL_AE_2step, self).__init__()
        self.num_dim = num_dim
        self.grid_bounds = grid_bounds
        self.likelihood = likelihood
        self.rho = rho
        self.ID = 20

        # Step 1: reduce the dimensionality of both the level of fidelity
        self.LF_ae0 = SVDKL_AE(num_dim, likelihood, grid_bounds, hidden_dim, grid_size, out_dim[0])
        self.HF_ae1 = SVDKL_AE(num_dim, likelihood, grid_bounds, hidden_dim, grid_size, out_dim[1])

    def forward(self, x_LF, x_HF):
        # Step 1: reduce the dimensionality of both the level of fidelity
        _, _, _, _, _, y_LF = self.LF_ae0(x_LF)
        _, _, _, _, _, y_HF = self.HF_ae1(x_HF)

        # Estimation of the instrinsic dimension
        ID_LF = eval_id(y_LF.detach().numpy())
        ID_HF = eval_id(y_HF.detach().numpy())
        self.ID = int(np.floor(np.max([ID_LF, ID_HF])))

        # Step 2: autoencoder with latent representation as input
        self.reducedEncoder_LF = reducedEncoder(self.ID)
        z_LF = self.reducedEncoder_LF(y_LF)

        self.reducedDecoder_LF = reducedDecoder(self.ID, self.num_dim)
        y_hat_LF = self.reducedDecoder_LF.decoder(z_LF)

        self.reducedEncoder_HF = reducedEncoder(self.ID)
        z_HF = self.reducedEncoder_HF(y_HF)

        self.reducedDecoder_HF = reducedDecoder(self.ID)
        z = z_HF + self.rho * z_LF
        y_hat_HF = self.reducedDecoder_HF.decoder(z)

        # Step 1: decoding on the outer level
        x_hat_LF, _ = self.LF_ae1.decoder(y_hat_LF)

        y = y_hat_HF + self.rho * y_hat_LF
        x_hat_HF, _ = self.HF_ae1.decoder(y)

        return x_hat_HF, z
