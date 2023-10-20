import torch
import numpy as np

from utils import sample_batch
from losses import loss_bce


def train(model, train_data, optimizer, batch_size, num_epochs):
    
    model.train()

    # if torch.backends.mps.is_available(): # on Apple Silicon
    #     mps_device = torch.device("mps")

    train_loss = 0

    sample_size = len(train_data)
    for i in range(int(sample_size/batch_size)):
        # Batch 
        data = np.array(sample_batch(train_data, batch_size))
        obs = torch.from_numpy(data).permute(0, 3, 1, 2)    # (batch_size, channels, height, width)
        
        # obs = obs.to(mps_device)  # uncomment this to pass data to Apple Silicon GPU (currently not working)

        # Initialization of the gradient
        optimizer.zero_grad()

        # Forward pass of the model
        mu_x, var_x, _, _, _, _ = model(obs)

        # Loss
        loss = loss_bce(mu_x, obs)

        # Gradient computation
        loss.backward()
        train_loss += loss.item()

        # Optimization step
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(num_epochs, train_loss / sample_size))



