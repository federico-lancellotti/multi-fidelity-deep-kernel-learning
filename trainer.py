import torch
import numpy as np

from utils import sample_batch
from losses import loss_bce


def train(model, train_data, optimizer, batch_size, num_epochs):
    model.train()

    # if torch.backends.mps.is_available(): # on Apple Silicon
    #     mps_device = torch.device("mps")

    train_loss = 0

    sample_size = train_data.size
    for i in range(int(sample_size / batch_size)):
        # Batch
        data, z_LF = train_data.sample_batch(batch_size)

        # Permutation to (batch_size, channels, height, width)
        obs = torch.from_numpy(data).permute(0, 3, 1, 2)

        # obs = obs.to(mps_device)  # uncomment this to pass data to Apple Silicon GPU (currently not working)

        # Initialization of the gradient
        optimizer.zero_grad()

        # Forward pass of the model
        mu_hat, var_hat, res, mean, covar, z = model(obs, z_LF)

        # Loss
        loss = loss_bce(mu_hat, obs)

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
