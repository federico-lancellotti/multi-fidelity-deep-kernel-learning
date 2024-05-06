import numpy as np
import torch


class DataLoader:
    """
    A class for loading and manipulating data for the multi-fidelity deep kernel learning model.

    Args:
        data (list): A list of dictionaries containing the data samples.
        z_LF (torch.Tensor): The latent representation of the data samples (at time t-1 and t) at the low-fidelity level.
        z_next_LF (torch.Tensor): The latent representation of the following data samples (at time t and t+1) at the low-fidelity level.
        z_fwd_LF (torch.Tensor): The latent representation produced by the forward part of the model at the low-fidelity level.
        obs_dim (tuple): The dimensions of the observation.

    Attributes:
        size (int): The number of data samples.
        obs (torch.Tensor): The tensor containing the observations, at time t-1 and t.
        next_obs (torch.Tensor): The tensor containing the next observations, at time t and t+1.
        z_LF (torch.Tensor): The latent representation of the data samples (at time t-1 and t) at the low-fidelity level.
        z_next_LF (torch.Tensor): The latent representation of the following data samples (at time t and t+1) at the low-fidelity level.
        z_fwd_LF (torch.Tensor): The latent representation produced by the forward part of the model at the low-fidelity level.
        state (torch.Tensor): The tensor containing the states, at time t-1 and t.
        next_state (torch.Tensor): The tensor containing the next states, at time t and t+1.
        done (torch.Tensor): The tensor indicating whether an episode is terminated at time t.

    Methods:
        sample_batch(batch_size=32): Returns a random batch of data samples.
        get_all_samples(): Returns all the samples and their low fidelity latent representations.
    """

    def __init__(self, data, z_LF, z_next_LF, z_fwd_LF, obs_dim):
        """
        Initializes the DataLoader object.

        Args:
            data (list): A list of dictionaries containing the data.
            z_LF (float): The z value for low fidelity.
            z_next_LF (float): The z value for next low fidelity.
            z_fwd_LF (float): The z value for forward low fidelity.
            obs_dim (tuple): A tuple representing the dimensions of the observations.

        Returns:
            None
        """

        self.size = len(data)
        self.obs = torch.zeros([int(self.size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])], dtype=torch.float32)
        self.next_obs = torch.zeros([int(self.size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])], dtype=torch.float32)
        self.z_LF = z_LF
        self.z_next_LF = z_next_LF
        self.z_fwd_LF = z_fwd_LF
        self.state = torch.zeros([int(self.size), data[0]["state"].shape[0]], dtype=torch.float32)
        self.next_state = torch.zeros([int(self.size), data[0]["next_state"].shape[0]], dtype=torch.float32)
        self.done = torch.zeros(int(self.size), dtype=bool)

        pos = 0
        for d in data:
            self.obs[pos] = torch.tensor(d["obs"], dtype=torch.float32) / 255
            self.next_obs[pos] = torch.tensor(d["next_obs"], dtype=torch.float32) / 255
            self.state[pos] = torch.tensor(d["state"])
            self.next_state[pos] = torch.tensor(d["next_state"])
            self.done[pos] = d["terminated"]
            pos += 1


    def sample_batch(self, batch_size=32):
        """
        Returns a random batch of data samples.

        Args:
            batch_size (int, optional): The size of the batch. Defaults to 32.

        Returns:
            dict: A dictionary containing the batch of data samples. In particular, the dictionary contains: 
                - obs: The observations, at time t-1 and t;
                - next_obs: The next observations, at time t and t+1;
                - z_LF: The latent representation of the data samples (at time t-1 and t) at the low-fidelity level;
                - z_next_LF: The latent representation of the following data samples (at time t and t+1) at the low-fidelity level;
                - z_fwd_LF: The latent representation produced by the forward part of the model at the low-fidelity level;
                - state: The states, at time t-1 and t;
                - next_state: The next states, at time t and t+1.
        """

        # Sample random indices
        idx = torch.randint(0, self.size, (batch_size,))

        # Adjust indices for episodes that are terminated
        for i in range(len(idx)):
            if self.done[i] == True:
                idx[i] = idx[i] - 1

        return dict(obs=self.obs[idx],
                    next_obs=self.next_obs[idx],
                    z_LF=self.z_LF[idx],
                    z_next_LF=self.z_next_LF[idx],
                    z_fwd_LF=self.z_fwd_LF[idx],
                    state=self.state[idx],
                    next_state=self.next_state[idx])


    def get_all_samples(self):
        """
        Returns all the samples and their low fidelity latent representations.

        Returns:
            dict: A dictionary containing all the samples and their latent representations. In particular, the dictionary contains: 
                - obs: The observations, at time t-1 and t;
                - next_obs: The next observations, at time t and t+1;
                - z_LF: The latent representation of the data samples (at time t-1 and t) at the low-fidelity level;
                - z_next_LF: The latent representation of the following data samples (at time t and t+1) at the low-fidelity level;
                - z_fwd_LF: The latent representation produced by the forward part of the model at the low-fidelity level;
                - state: The states, at time t-1 and t;
                - next_state: The next states, at time t and t+1.
        """

        return dict(obs=self.obs,
                    next_obs=self.next_obs,
                    z_LF=self.z_LF,
                    z_next_LF=self.z_next_LF,
                    z_fwd_LF=self.z_fwd_LF,
                    state=self.state,
                    next_state=self.next_state)
