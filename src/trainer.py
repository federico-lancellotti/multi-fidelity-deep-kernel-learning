import torch
import numpy as np

from losses import loss_bce, kl_divergence_balance


def train(
    model,
    train_data,
    optimizers,
    batch_size,
    num_epochs,
    variational_kl_term,
    variational_kl_term_fwd,
    k1=1,
    beta=1,
    use_gpu=False,
    flush=False,
):
    """
    Trains the model using the given training data.

    Args:
        model (object): The model to be trained.
        train_data (object): The training data.
        optimizers (list): A list of optimizers for each component of the model.
        batch_size (int): The batch size for training.
        num_epochs (int): The number of epochs to train for.
        variational_kl_term (function): The function to compute the variational KL term.
        variational_kl_term_fwd (function): The function to compute the variational KL term for the forward model.
        k1 (int, optional): The value of k1. Defaults to 1.
        beta (int, optional): The value of beta. Defaults to 1.
    """

    model.train()
    model.AE_DKL.likelihood.train()
    model.fwd_model_DKL.likelihood.train()

    device = torch.device("cpu")
    if use_gpu: 
        if torch.cuda.is_available():
            device = torch.device("cuda")
    #     if torch.backends.mps.is_available(): # on Apple Silicon
    #         mps_device = torch.device("mps")

    optimizer = optimizers[0]       # optimizer for the model
    optimizer_var1 = optimizers[1]  # optimizer for the variational term
    optimizer_var2 = optimizers[2]  # optimizer for the variational term for the forward model

    # Initialize the losses
    train_loss = 0
    train_loss_vae = 0
    train_loss_varKL_vae = 0
    # train_loss_fwd_rec = 0
    train_loss_fwd_res = 0
    train_loss_varKL_fwd = 0

    sample_size = train_data.size
    for i in range(int(sample_size / batch_size)):
        # Batch
        batch, _ = train_data.sample_batch(batch_size)

        # Permutation to (batch_size, channels, height, width)
        obs = batch["obs"].permute(0, 3, 1, 2).to(device)
        next_obs = batch["next_obs"].permute(0, 3, 1, 2).to(device)
        z_LF = batch["z_LF"].to(device)
        z_next_LF = batch["z_next_LF"].to(device)
        z_fwd_LF = batch["z_fwd_LF"].to(device)

        # Initialization of the gradients
        optimizer.zero_grad()
        optimizer_var1.zero_grad()
        optimizer_var2.zero_grad()

        # Forward pass of the model
        mu_x, var_x, _, _, _, res, mu_target, var_target, _, res_target, mu_fwd, _, res_fwd, _, mu_x_fwd = model(obs, z_LF, next_obs, z_next_LF, z_fwd_LF)

        # Loss
        loss_vae = loss_bce(mu_x, obs)
        loss_varKL_vae = variational_kl_term(beta=1)

        # compute forward model loss (KL divergence) + variational inference
        # loss_fwd_rec = loss_bce(mu_x_fwd, next_obs)
        loss_fwd_res = -beta * kl_divergence_balance(
            model.AE_DKL.likelihood(res_target).mean,
            model.AE_DKL.likelihood(res_target).variance,
            model.fwd_model_DKL.likelihood(res_fwd).mean,
            model.fwd_model_DKL.likelihood(res_fwd).variance,
            alpha=0.8,
            dim=1,
        )
        loss_varKL_fwd = variational_kl_term_fwd(beta=1)

        # loss = loss_vae + loss_fwd_rec - loss_fwd_res
        loss = loss_vae - loss_fwd_res

        loss_varKL_v = -loss_varKL_vae
        loss_varKL_f = -loss_varKL_fwd

        # Gradient computation
        loss.backward(retain_graph=True)
        loss_varKL_v.backward()
        loss_varKL_f.backward()

        train_loss += loss.item()
        train_loss_vae += loss_vae.item()
        train_loss_varKL_vae += loss_varKL_v.item()
        # train_loss_fwd_rec += loss_fwd_rec.item()
        train_loss_fwd_res += -loss_fwd_res.item()
        train_loss_varKL_fwd += loss_varKL_f.item()

        # Optimization step
        optimizer.step()
        optimizer_var1.step()
        optimizer_var2.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(num_epochs, train_loss / sample_size), flush=flush)
    print('====> Epoch: {} Average VAE loss: {:.4f}'.format(num_epochs, train_loss_vae / sample_size), flush=flush)
    print('====> Epoch: {} Average variational loss: {:.4f}'.format(num_epochs, train_loss_varKL_vae / sample_size), flush=flush)
    # print('====> Epoch: {} Average FWD reconstruction loss: {:.4f}'.format(num_epochs, train_loss_fwd_rec / sample_size), flush=flush)
    print('====> Epoch: {} Average FWD residuals loss: {:.4f}'.format(num_epochs, train_loss_fwd_res / sample_size), flush=flush)
    print('====> Epoch: {} Average FWD variational loss: {:.4f}'.format(num_epochs, train_loss_varKL_fwd / sample_size), flush=flush)
