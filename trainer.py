import torch
import numpy as np

from utils import sample_batch
from losses import loss_bce, loss_negloglikelihood, kl_divergence_balance


def train_recon(model, train_data, optimizer, batch_size, num_epochs):
    model.train()

    # if torch.backends.mps.is_available(): # on Apple Silicon
    #     mps_device = torch.device("mps")

    train_loss = 0

    sample_size = train_data.size
    for i in range(int(sample_size / batch_size)):
        # Batch
        batch = train_data.sample_batch(batch_size)

        # Permutation to (batch_size, channels, height, width)
        batch["obs"] = torch.from_numpy(batch["obs"]).permute(0, 3, 1, 2)
        # obs = obs.to(mps_device)  # uncomment this to pass data to Apple Silicon GPU (currently not working)

        # Initialization of the gradient
        optimizer.zero_grad()

        # Forward pass of the model
        mu_hat, var_hat, res, mean, covar, z = model(batch["obs"], batch["z_LF"])

        # Loss
        loss = loss_bce(mu_hat, batch["obs"])

        # Gradient computation
        loss.backward()
        train_loss += loss.item()

        # Optimization step
        optimizer.step()

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            num_epochs, train_loss / sample_size
        )
    )


def train_dyn(
    model,
    train_data,
    optimizers,
    batch_size,
    num_epochs,
    variational_kl_term,
    variational_kl_term_fwd,
    k1=1,
    beta=1,
):
    model.train()
    model.AE_DKL.likelihood.train()
    model.fwd_model_DKL.likelihood.train()

    # if torch.backends.mps.is_available(): # on Apple Silicon
    #     mps_device = torch.device("mps")

    optimizer = optimizers[0]
    optimizer_var1 = optimizers[1]
    optimizer_var2 = optimizers[2]

    train_loss = 0
    train_loss_vae = 0
    train_loss_varKL_vae = 0
    train_loss_fwd = 0
    train_loss_varKL_fwd = 0

    sample_size = train_data.size
    for i in range(int(sample_size / batch_size)):
        # Batch
        batch = train_data.sample_batch(batch_size)

        # Permutation to (batch_size, channels, height, width)
        obs = torch.from_numpy(batch["obs"]).permute(0, 3, 1, 2)
        next_obs = torch.from_numpy(batch["next_obs"]).permute(0, 3, 1, 2)
        # obs = obs.to(mps_device)  # uncomment this to pass data to Apple Silicon GPU (currently not working)

        # Initialization of the gradients
        optimizer.zero_grad()
        optimizer_var1.zero_grad()
        optimizer_var2.zero_grad()

        # Forward pass of the model
        (
            mu_x,
            var_x,
            _,
            _,
            _,
            res,
            mu_target,
            var_target,
            _,
            res_target,
            _,
            _,
            res_fwd,
            _,
        ) = model(obs, batch["z_LF"], next_obs, batch["z_next_LF"], batch["z_fwd_LF"])

        # Loss
        loss_vae = loss_bce(mu_x, obs)
        loss_varKL_vae = variational_kl_term(beta=1)

        # compute forward model loss (KL divergence) + variational inference
        loss_fwd = -beta * kl_divergence_balance(
            model.AE_DKL.likelihood(res_target).mean,
            model.AE_DKL.likelihood(res_target).variance,
            model.fwd_model_DKL.likelihood(res_fwd).mean,
            model.fwd_model_DKL.likelihood(res_fwd).variance,
            alpha=0.8,
            dim=1,
        )
        loss_varKL_fwd = variational_kl_term_fwd(beta=1)

        loss = loss_vae - loss_fwd

        loss_varKL_v = -loss_varKL_vae
        loss_varKL_f = -loss_varKL_fwd

        # Gradient computation
        loss.backward(retain_graph=True)
        loss_varKL_v.backward()
        loss_varKL_f.backward()

        train_loss += loss.item()
        train_loss_vae += loss_vae.item()
        train_loss_varKL_vae += loss_varKL_v.item()
        train_loss_fwd += -loss_fwd.item()
        train_loss_varKL_fwd += loss_varKL_f.item()

        # Optimization step
        optimizer.step()
        optimizer_var1.step()
        optimizer_var2.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(num_epochs, train_loss / sample_size))
    print('====> Epoch: {} Average VAE loss: {:.4f}'.format(num_epochs, train_loss_vae / sample_size))
    print('====> Epoch: {} Average variational loss: {:.4f}'.format(num_epochs, train_loss_varKL_vae / sample_size))
    print('====> Epoch: {} Average FWD loss: {:.4f}'.format(num_epochs, train_loss_fwd / sample_size))
    print('====> Epoch: {} Average FWD variational loss: {:.4f}'.format(num_epochs, train_loss_varKL_fwd / sample_size))
    