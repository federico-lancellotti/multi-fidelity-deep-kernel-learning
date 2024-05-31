import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import math


OUT_DIM = {128: 57, 100: 43, 84: 35, 64: 25, 42: 14, 32: 9, 24: 5}


class Encoder(nn.Module):
    """
    Encoder module for the multi-fidelity DKL model.

    The encoder is composed of 4 convolutional layers with 32 filters per layer. 
    The convolutional filters are of size (3x3) and shifted across the images with stride 1 
    (only the first convolutional layer has stride 2 to quickly reduce the input dimensionality). 
    Batch normalization is also used after the 2nd and 4th convolutional layers. 
    The output features of the last convolutional layer are flattened and fed to two final fully 
    connected layers of dimensions 256 and 20, respectively, compressing the features 
    to a 20-dimensional feature vector. 
    Each layer has ELU activations, except the last fully-connected layer with a linear activation.

    Args:
        hidden_dim (int): The dimensionality of the hidden layer.
        z_dim (int): The dimensionality of the latent space.
        out_dim (int): The output dimensionality.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        batch1 (nn.BatchNorm2d): The first batch normalization layer.
        conv3 (nn.Conv2d): The third convolutional layer.
        conv4 (nn.Conv2d): The fourth convolutional layer.
        batch2 (nn.BatchNorm2d): The second batch normalization layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
    """

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
        """
        Encodes the input tensor into the latent space.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The encoded tensor in the latent space.
        """

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
        """
        Forward pass of the encoder. Calls the encoder method.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The encoded tensor in the latent space.
        """

        return self.encoder(x)


# The decoder is composed of a linear fully-connected layer and 4 transpose
# convolutional layers with 32 filters each. The convolutional filters are of
# size (3 × 3) and shifted across the images with stride 1 (again, the last
# convolutional layer has stride 2). Batch normalization is used after the
# 2nd and 4th convolutional layers, and ELU activations are employed for all
# the layers except the last one.
# The outputs are the mean μ_xhat_t and variance sigma2_xhat_t of N(mu_xhat_t, sigma2_xhat_t).
class Decoder(nn.Module):
    """
    Decoder module for generating output from latent space.

    The decoder is composed of a linear fully-connected layer and 4 transpose 
    convolutional layers with 32 filters each.
    The convolutional filters are of size (3x3) and shifted across the images with 
    stride 1 (the last convolutional layer has stride 2).
    Batch normalization is used after the 2nd and 4th convolutional layers, and ELU 
    activations are employed for all the layers except the last one.
    The outputs are the mean μ_xhat_t and variance sigma2_xhat_t of N(μ_xhat_t, sigma_xhat_t).

    Args:
        z_dim (int): Dimension of the latent space. Default is 20.
        out_dim (int): Dimension of the output. Default is 84.
    """

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
        """
        Decodes the latent space representation back to the original space, 
        producing a reconstruction of the input image.

        Args:
            z (torch.Tensor): Latent space representation.

        Returns:
            mu (torch.Tensor): Mean of the generated output.
            std (torch.Tensor): Standard deviation of the generated output.
        """

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
        """
        Forward pass of the decoder. Calls the decoder method.

        Args:
            z (torch.Tensor): Latent space representation.

        Returns:
            torch.Tensor: Generated output.
        """

        return self.decoder(z)


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    """
    Gaussian Process Layer class.

    This class represents a Gaussian Process (GP) layer in a multi-layer deep kernel learning (DKL) model.
    It extends the `ApproximateGP` class from the `gpytorch.models` module.

    Args:
        num_dim (int): Number of dimensions (tasks) in the GP model, namely the dimensionality of the output space.
        grid_size (int, optional): The grid size used to approximate the GP model. A larger grid size provides a denser approximation. Default is 64.
        grid_bounds (tuple, optional): Specifies the bounds of the regularization. Default is (-10.0, 10).

    Attributes:
        mean_module (gpytorch.means.ConstantMean): Mean module of the GP model.
        covar_module (gpytorch.kernels.ScaleKernel): Covariance module of the GP model.
        grid_bounds (tuple): Bounds of the regularization.

    """

    def __init__(self, num_dim, grid_size=64, grid_bounds=(-10.0, 10)):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

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
        super().__init__(variational_strategy)

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
        """
        Forward pass of the Gaussian Process Layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            gpytorch.distributions.MultivariateNormal: Multivariate normal distribution representing the output of the GP layer.

        """

        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class SVDKL_AE(gpytorch.Module):
    """
    Variational Autoencoder with Stochastic Variational Deep Kernel Learning (SVDKL) architecture.

    Args:
        num_dim (int): Number of input dimensions.
        likelihood: Likelihood function for the Gaussian process.
        grid_bounds (tuple, optional): Bounds of the grid for the Gaussian process. Defaults to (-10.0, 10.0).
        hidden_dim (int, optional): Dimension of the hidden layer in the encoder. Defaults to 32.
        grid_size (int, optional): Size of the grid for the Gaussian process. Defaults to 32.
        obs_dim (int, optional): Dimension of the observation. Defaults to 84.
        rho (int, optional): Scaling factor for the low-fidelity input. Defaults to 1.
        num_dim_LF (int, optional): Number of dimensions for the low-fidelity input. Defaults to 0.

    Attributes:
        num_dim (int): Number of input dimensions.
        out_dim (int): Number of output dimensions.
        rho (int): Scaling factor for the low-fidelity input.
        grid_bounds (tuple): Bounds of the grid for the Gaussian process.
        likelihood: Likelihood function for the Gaussian process.
        num_dim_LF (int): Number of dimensions for the low-fidelity input.
        gp_layer: Gaussian process layer.
        encoder: Encoder module.
        decoder: Decoder module.
        scale_to_bounds: Module to scale the neural network features to the grid bounds.
        fc_LF: Linear layer to transform the low-fidelity input if the number of dimensions is different.

    """

    def __init__(
        self,
        num_dim,
        likelihood,
        grid_bounds=(-10.0, 10.0),
        hidden_dim=32,
        grid_size=32,
        obs_dim=84,
        rho=1,
        num_dim_LF=0,
    ):
        super(SVDKL_AE, self).__init__()
        self.num_dim = num_dim
        self.out_dim = OUT_DIM[obs_dim]
        self.rho = rho
        self.grid_bounds = grid_bounds
        self.likelihood = likelihood
        self.num_dim_LF = num_dim_LF if num_dim_LF else num_dim

        self.gp_layer = GaussianProcessLayer(num_dim, grid_size, grid_bounds)
        self.encoder = Encoder(hidden_dim, self.num_dim, self.out_dim)
        self.decoder = Decoder(self.num_dim, self.out_dim)

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            self.grid_bounds[0], self.grid_bounds[1]
        )

        if self.num_dim_LF != self.num_dim:
            self.fc_LF = nn.Linear(self.num_dim_LF, self.num_dim)


    def forward(self, x, z_LF):
        """
        Forward pass of the SVDKL-AE model. 
        The input data is passed through the encoder, and the latent space representation 
        is added to the latent space representation of the low-fidelity input.
        The combined latent space representation is then passed through the Gaussian process layer.
        The output of the Gaussian process layer is passed through the likelihood function to obtain 
        the mean and variance.
        The mean and variance are then passed through the decoder to obtain the reconstructed output.

        Args:
            x: Input data.
            z_LF: Low-fidelity input data.

        Returns:
            mu_hat: Mean of the decoder output.
            var_hat: Variance of the decoder output.
            res: Gaussian process result.
            mean: Mean of the Gaussian process result.
            covar: Covariance of the Gaussian process result.
            z: Sampled latent variable.

        """
        if self.num_dim_LF != self.num_dim:
            z_LF = self.fc_LF(z_LF)

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

    
class SVDKL_AE_latent_dyn(nn.Module):
    """
    A class representing the model for learning the dynamics of a system using the Stochastic Variational 
    Deep Kernel Learning Autoencoder architecture.

    The first part of the model is an SVDKL Autoencoder that learns the latent space representation of the input data.
    The second part of the model is a DKL forward model that learns the dynamics of the latent space representation.

    Args:
        num_dim (int): The number of latent dimensions.
        likelihood (torch.distributions.Distribution): The likelihood distribution for the autoencoder.
        likelihood_fwd (torch.distributions.Distribution): The likelihood distribution for the forward model.
        grid_bounds (tuple, optional): The bounds of the grid. Defaults to (-10., 10.).
        h_dim (int, optional): The hidden dimension size. Defaults to 32.
        grid_size (int, optional): The size of the grid. Defaults to 32.
        obs_dim (int, optional): The dimension of the observations. Defaults to 84.
        rho (int, optional): The value of rho. Defaults to 1.
        num_dim_LF (int, optional): The number of latent dimensions for low-fidelity. Defaults to 0.
    """

    def __init__(self, num_dim, likelihood, likelihood_fwd, grid_bounds=(-10., 10.), h_dim=32, grid_size=32, obs_dim=84, rho=1, num_dim_LF=0):
        """
        Initialize the SVDKL_AE_latent_dyn class.

        Args:
            num_dim (int): The number of dimensions.
            likelihood (str): The likelihood function to use.
            likelihood_fwd (str): The likelihood function for the forward model.
            grid_bounds (tuple, optional): The bounds of the grid. Defaults to (-10., 10.).
            h_dim (int, optional): The hidden dimension. Defaults to 32.
            grid_size (int, optional): The size of the grid. Defaults to 32.
            obs_dim (int, optional): The observation dimension. Defaults to 84.
            rho (int, optional): The value of rho. Defaults to 1.
            num_dim_LF (int, optional): The number of dimensions for the low-fidelity model. Defaults to 0.
        """
        
        super(SVDKL_AE_latent_dyn, self).__init__()

        self.obs_dim = obs_dim
        self.num_dim = num_dim
        self.num_dim_LF = num_dim_LF

        self.AE_DKL = SVDKL_AE(num_dim=num_dim, likelihood=likelihood, grid_bounds=grid_bounds, hidden_dim=h_dim, 
                               grid_size=grid_size, obs_dim=obs_dim, rho=rho, num_dim_LF=num_dim_LF)
        self.fwd_model_DKL = Forward_DKLModel(num_dim=num_dim, grid_bounds=grid_bounds, h_dim=h_dim,
                                              grid_size=grid_size, likelihood=likelihood_fwd, rho=rho, num_dim_LF=num_dim_LF)  # DKL forward model

    def forward(self, x, z_LF, x_next, z_next_LF, z_fwd_LF):
        """
        Forward pass of the SVDKL_AE_latent_dyn model.

        Args:
            x (torch.Tensor): Input data.
            z_LF (torch.Tensor): Latent space representation of the low-fidelity input data.
            x_next (torch.Tensor): Next input data.
            z_next_LF (torch.Tensor): Latent space representation of the low-fidelity next input data.
            z_fwd_LF (torch.Tensor): Latent space representation of the low-fidelity forward data.

        Returns:
            tuple: A tuple containing the following elements:
                - mu_x (torch.Tensor): Mean of the reconstructed input data.
                - var_x (torch.Tensor): Variance of the reconstructed input data.
                - mu (torch.Tensor): Mean of the latent space representation.
                - var (torch.Tensor): Variance of the latent space representation.
                - z (torch.Tensor): Latent space representation.
                - res (torch.Tensor): Residuals.
                - mu_target (torch.Tensor): Mean of the reconstructed next input data.
                - var_target (torch.Tensor): Variance of the reconstructed next input data.
                - z_target (torch.Tensor): Latent space representation of the next input data.
                - res_target (torch.Tensor): Residuals of the next input data.
                - mu_fwd (torch.Tensor): Mean of the latent space representation for the forward model.
                - var_fwd (torch.Tensor): Variance of the latent space representation for the forward model.
                - res_fwd (torch.Tensor): Residuals for the forward model.
                - z_fwd (torch.Tensor): Latent space representation for the forward model.
                - mu_x_rec (torch.Tensor): Mean of the reconstructed input data from the forward model.
        """

        mu_x, var_x, res, mu, var, z = self.AE_DKL(x, z_LF)
        mu_x_target, var_x_target, res_target, mu_target, var_target, z_target = self.AE_DKL(x_next, z_next_LF)
        
        res_fwd, mu_fwd, var_fwd, z_fwd = self.fwd_model_DKL(z, z_fwd_LF)
        mu_x_rec, _ = self.AE_DKL.decoder(z_fwd)

        return mu_x, var_x, mu, var, z, res, mu_target, var_target, z_target, res_target, mu_fwd, var_fwd, res_fwd, z_fwd, mu_x_rec

    def predict_dynamics(self, z, z_fwd_LF, samples=1):
        """
        Predict the dynamics of the system.

        Args:
            z (torch.Tensor): Latent space representation.
            z_fwd_LF (torch.Tensor): Latent space representation of the low-fidelity forward data.
            samples (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            tuple: A tuple containing the following elements:
                - mu_x_rec (torch.Tensor): Mean of the reconstructed input data from the forward model.
                - z_fwd (torch.Tensor): Latent space representation for the forward model.
                - mu_fwd (torch.Tensor): Mean of the latent space representation for the forward model.
                - res_fwd (torch.Tensor): Residuals for the forward model.
        """

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
        """
        Predict the mean dynamics of the system.

        Args:
            mu (torch.Tensor): Mean of the latent space representation.
            z_fwd_LF (torch.Tensor): Latent space representation of the low-fidelity forward data.

        Returns:
            tuple: A tuple containing the following elements:
                - mu_x_rec (torch.Tensor): Mean of the reconstructed input data from the forward model.
                - z_fwd (torch.Tensor): Latent space representation for the forward model.
                - mu_fwd (torch.Tensor): Mean of the latent space representation for the forward model.
                - res_fwd (torch.Tensor): Residuals for the forward model.
        """

        res_fwd, mu_fwd, var_fwd, z_fwd = self.fwd_model_DKL(mu, z_fwd_LF)
        mu_x_rec, _ = self.AE_DKL.decoder(mu_fwd)
        return mu_x_rec, z_fwd, mu_fwd, res_fwd
    

class Forward_DKLModel(gpytorch.Module):
    """
    Forward_DKLModel is a class that represents a forward deep kernel learning model.
    It learns the forward dynamics of the latent space representation.

    The input data is passed through a neural network model, and the latent space representation 
    is added to the latent space representation of the low-fidelity input.

    Args:
        num_dim (int): The number of input dimensions.
        likelihood: The likelihood function for the Gaussian process.
        grid_bounds (tuple, optional): The bounds of the grid. Defaults to (-10., 10.).
        h_dim (int, optional): The hidden dimension size for the neural network model. Defaults to 256.
        grid_size (int, optional): The size of the grid. Defaults to 32.
        rho (int, optional): The scaling factor for the low-fidelity input. Defaults to 1.
        num_dim_LF (int, optional): The number of dimensions for the low-fidelity input. Defaults to 0.

    Attributes:
        gp_layer_2: The Gaussian process layer.
        grid_bounds (tuple): The bounds of the grid.
        num_dim (int): The number of input dimensions.
        likelihood: The likelihood function for the Gaussian process.
        rho (int): The scaling factor for the low-fidelity input.
        num_dim_LF (int): The number of dimensions for the low-fidelity input.
        fwd_model: The forward neural network model.
        scale_to_bounds: The module that scales the neural network features to nice values.
        fc_LF: The linear layer for low-fidelity input transformation.

    Methods:
        forward: Performs the forward pass of the model.

    """

    def __init__(self, num_dim, likelihood, grid_bounds=(-10., 10.), h_dim=256, grid_size=32, rho=1, num_dim_LF=0):
        super(Forward_DKLModel, self).__init__()
        self.gp_layer_2 = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds, grid_size=grid_size)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim
        self.likelihood = likelihood
        self.rho = rho
        self.num_dim_LF = num_dim_LF if num_dim_LF else num_dim

        self.fwd_model = ForwardModel(num_dim, h_dim) # NN model

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

        if self.num_dim_LF != self.num_dim:
            self.fc_LF = nn.Linear(self.num_dim_LF, self.num_dim)

    def forward(self, x, z_LF):
        """
        Performs the forward pass of the model.

        Args:
            x: The input data.
            z_LF: The low-fidelity input data.

        Returns:
            res: The output of the Gaussian process layer.
            mean: The mean of the Gaussian process output.
            var: The variance of the Gaussian process output.
            z: The sampled output from the likelihood function.

        """

        if self.num_dim_LF != self.num_dim:
            z_LF = self.fc_LF(z_LF)

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
    """
    A forward model that maps input z to output features, providing a representation of the following frame.

    Args:
        z_dim (int): The dimensionality of the input z.
        h_dim (int): The dimensionality of the hidden layer.

    Attributes:
        fc (nn.Linear): The fully connected layer from z_dim to h_dim.
        fc1 (nn.Linear): The fully connected layer from h_dim to h_dim.
        fc2 (nn.Linear): The fully connected layer from h_dim to z_dim.
        batch (nn.BatchNorm1d): Batch normalization layer for z_dim.
    """

    def __init__(self, z_dim=20, h_dim=256):
        super(ForwardModel, self).__init__()

        self.fc = nn.Linear(z_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.batch = nn.BatchNorm1d(z_dim)

    def forward(self, z):
        """
        Forward pass of the forward model.

        Args:
            z (torch.Tensor): The input tensor of shape (batch_size, z_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, z_dim).

        """
        
        z = F.elu(self.fc(z))
        z = F.elu(self.fc1(z))
        features = self.fc2(z)
        return features