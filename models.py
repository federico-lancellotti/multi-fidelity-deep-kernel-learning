import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import math

# for 84x84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}


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
    def __init__(self, hidden_dim=256, z_dim=20):
        super(Encoder, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=(3, 3), stride=(2, 2))    # stride 2 to quickly reduce the input dimensionality
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))   # (first) input shape (84,84,6) -> output shape (35,35,32)
        self.batch2 = nn.BatchNorm2d(32)
        out_dim = OUT_DIM[4]
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
    def __init__(self, z_dim=20):
        super(Decoder, self).__init__()
    
        # decoder part
        out_dim = OUT_DIM[4]
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
    def __init__(self, num_dim, grid_size=64, grid_bounds=(-10.,10)):
        # num_dim: number of dimensions (tasks) in the GP model, namely the dimensionality of the output space.
        # grid_size: the grid is a set of points used to approximate the GP model. A larger grid size provides a denser approximation.
        # grid_bounds: specifies the bounds of the grid in the input space.
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )
        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)  # calls the constructor of the base class (gpytorch.models.ApproximateGP) and initializes the GP model with the specified variational strategy

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=gpytorch.kernels.RBFKernel(
                ard_num_dims=num_dim, 
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
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
    def __init__(self, num_dim, likelihood, grid_bounds=(-10.,10.), hidden_dim=32, grid_size=32):
        super(SVDKL_AE, self).__init__()
        self.num_dim = num_dim
        self.grid_bounds = grid_bounds
        self.likelihood = likelihood

        self.gp_layer = GaussianProcessLayer(num_dim, grid_size, grid_bounds)
        self.encoder = Encoder(hidden_dim, self.num_dim)
        self.decoder = Decoder(self.num_dim)

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x):
        features = self.encoder(x)
        features = self.scale_to_bounds(features)
        features = features.transpose(-1, -2).unsqueeze(-1) # This line makes it so that we learn a GP for each feature
        
        if self.training:
            with gpytorch.settings.detach_test_caches(False):
                self.gp_layer.train()
                self.gp_layer.eval()
                res = self.gp_layer(features)
        else:
            res = self.gp_layer(features)

        mean = res.mean
        covar = res.covar
        z = self.likelihood(res).rsample()

        mu_hat, var_hat = self.decoder.decoder(z)

        return mu_hat, var_hat, res, mean, covar, z
