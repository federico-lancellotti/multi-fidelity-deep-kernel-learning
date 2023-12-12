import torch
import numpy as np

from utils import sample_batch
from losses import loss_bce


def train(model, train_data, optimizer, batch_size, num_epochs):
    model.train()

    # if torch.backends.mps.is_available(): # on Apple Silicon
    #     mps_device = torch.device("mps")

    train_loss = 0

    sample_size = train_data[0].size
    for i in range(int(sample_size / batch_size)):
        # Batch
        data = [
            np.array(train_data[0].sample_batch(batch_size)),
            np.array(train_data[1].sample_batch(batch_size)),
        ]

        # Permutation to (batch_size, channels, height, width)
        obs = [
            torch.from_numpy(data[0]).permute(0, 3, 1, 2),
            torch.from_numpy(data[1]).permute(0, 3, 1, 2),
        ]

        # obs = obs.to(mps_device)  # uncomment this to pass data to Apple Silicon GPU (currently not working)

        # Initialization of the gradient
        optimizer.zero_grad()

        # Forward pass of the model
        # mu_x_LF, var_x_LF, mu_x_HF, var_x_HF, res_HF, mean_HF, covar_HF, z = model(
        #     obs[0], obs[1]
        # )
        mu_x_HF, z = model(obs[0], obs[1])

        # Loss
        loss = loss_bce(mu_x_HF, obs[1])
        # loss = loss_bce(mu_x_LF, obs[0]) + loss_bce(mu_x_HF, obs[1])

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
