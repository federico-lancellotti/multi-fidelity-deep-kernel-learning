import torch
from .utils import check_indices


class BaseDataLoader:
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
        done (torch.Tensor): The tensor indicating whether an episode is terminated at time t.

    Methods:
        sample_batch(batch_size=32): Returns a random batch of data samples.
        sample_batch_with_idx(idx): Returns a batch of data samples with the given indices.
        get_all_samples(): Returns all the samples and their low fidelity latent representations.

    """

    def __init__(self, data, z_LF, z_next_LF, z_fwd_LF, obs_dim):
        """
        Initializes the BaseDataLoader object.

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
        self.done = torch.zeros(int(self.size), dtype=bool)

        pos = 0
        for d in data:
            self.obs[pos] = torch.tensor(d["obs"], dtype=torch.float32) / 255.0
            self.next_obs[pos] = torch.tensor(d["next_obs"], dtype=torch.float32) / 255.0
            self.done[pos] = d["terminated"]
            pos += 1


    def sample_batch(self, batch_size=32):
        """
        Returns a random batch of data samples.

        Args:
            batch_size (int, optional): The size of the batch. Defaults to 32.

        Returns:
            tuple:
                - batch (dict): A dictionary containing the batch of data samples. In particular, the dictionary contains: 
                    - obs: The observations, at time t-1 and t;
                    - next_obs: The next observations, at time t and t+1;
                    - z_LF: The latent representation of the data samples (at time t-1 and t) at the low-fidelity level;
                    - z_next_LF: The latent representation of the following data samples (at time t and t+1) at the low-fidelity level;
                    - z_fwd_LF: The latent representation produced by the forward part of the model at the low-fidelity level.
                - idx (torch.Tensor): The indices of the samples in the batch.
        """

        # Sample random indices
        idx = torch.randint(0, self.size, (batch_size,))

        # Adjust indices for episodes that are terminated
        for i in range(len(idx)):
            if self.done[i] == True:
                idx[i] = idx[i] - 1

        # # Check if the indices are within bounds
        # check_indices(self.obs, idx)
        # check_indices(self.next_obs, idx)
        # check_indices(self.z_LF, idx)
        # check_indices(self.z_next_LF, idx)
        # check_indices(self.z_fwd_LF, idx)

        # Create the batch
        batch = dict(obs=self.obs[idx],
                        next_obs=self.next_obs[idx],
                        z_LF=self.z_LF[idx],
                        z_next_LF=self.z_next_LF[idx],
                        z_fwd_LF=self.z_fwd_LF[idx])

        return batch, idx


    def sample_batch_with_idx(self, idx):
        """
        Returns a batch of data samples with the given indices.

        Args:
            idx (list of int): The indices of the samples to be included in the batch.

        Returns:
            dict: A dictionary containing the batch of data samples. In particular, the dictionary contains: 
                - obs: The observations, at time t-1 and t;
                - next_obs: The next observations, at time t and t+1;
                - z_LF: The latent representation of the data samples (at time t-1 and t) at the low-fidelity level;
                - z_next_LF: The latent representation of the following data samples (at time t and t+1) at the low-fidelity level;
                - z_fwd_LF: The latent representation produced by the forward part of the model at the low-fidelity level.
        """

        return dict(obs=self.obs[idx],
                    next_obs=self.next_obs[idx],
                    z_LF=self.z_LF[idx],
                    z_next_LF=self.z_next_LF[idx],
                    z_fwd_LF=self.z_fwd_LF[idx])


    def get_all_samples(self):
        """
        Returns all the samples and their low fidelity latent representations.

        Returns:
            dict: A dictionary containing all the samples and their latent representations. In particular, the dictionary contains: 
                - obs: The observations, at time t-1 and t;
                - next_obs: The next observations, at time t and t+1;
                - z_LF: The latent representation of the data samples (at time t-1 and t) at the low-fidelity level;
                - z_next_LF: The latent representation of the following data samples (at time t and t+1) at the low-fidelity level;
                - z_fwd_LF: The latent representation produced by the forward part of the model at the low-fidelity level.
        """

        return dict(obs=self.obs,
                    next_obs=self.next_obs,
                    z_LF=self.z_LF,
                    z_next_LF=self.z_next_LF,
                    z_fwd_LF=self.z_fwd_LF)


class GymDataLoader(BaseDataLoader):
    """
    A class for loading and manipulating data for the multi-fidelity deep kernel learning model, using data from the OpenAI Gym environment.
    It inherits from the BaseDataLoader class.

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
        state (torch.Tensor): The tensor containing the states, at time t.
        next_state (torch.Tensor): The tensor containing the next states, at time t+1.
        done (torch.Tensor): The tensor indicating whether an episode is terminated at time t.

    Methods:
        sample_batch(batch_size=32): Returns a random batch of data samples.
        sample_batch_with_idx(idx): Returns a batch of data samples with the given indices.
        get_all_samples(): Returns all the samples and their low fidelity latent representations.

    """
    
    def __init__(self, data, z_LF, z_next_LF, z_fwd_LF, obs_dim):
        """
        Initializes the GymDataLoader object.

        Args:
            data (list): A list of dictionaries containing the data.
            z_LF (float): The z value for low fidelity.
            z_next_LF (float): The z value for next low fidelity.
            z_fwd_LF (float): The z value for forward low fidelity.
            obs_dim (tuple): A tuple representing the dimensions of the observations.

        Returns:
            None
        """

        super().__init__(data, z_LF, z_next_LF, z_fwd_LF, obs_dim)
        self.state = torch.zeros([int(self.size), data[0]["state"].shape[0]], dtype=torch.float32)
        self.next_state = torch.zeros([int(self.size), data[0]["next_state"].shape[0]], dtype=torch.float32)

        pos = 0
        for d in data:
            self.state[pos] = torch.tensor(d["state"])
            self.next_state[pos] = torch.tensor(d["next_state"])
            pos += 1


    def sample_batch(self, batch_size=32):
        """
        Returns a random batch of data samples.

        Args:
            batch_size (int, optional): The size of the batch. Defaults to 32.

        Returns:
            tuple:
                - dict: A dictionary containing the batch of data samples. The dictionary contains:
                    - obs (torch.Tensor): The observations, at time t-1 and t.
                    - next_obs (torch.Tensor): The next observations, at time t and t+1.
                    - z_LF (torch.Tensor): The latent representation of the data samples (at time t-1 and t) at the low-fidelity level.
                    - z_next_LF (torch.Tensor): The latent representation of the following data samples (at time t and t+1) at the low-fidelity level.
                    - z_fwd_LF (torch.Tensor): The latent representation produced by the forward part of the model at the low-fidelity level.
                    - state (torch.Tensor): The state at time t.
                    - next_state (torch.Tensor): The state at time t+1.
                - torch.Tensor: The indices of the samples in the batch.
        """

        # Sample a batch
        batch, idx = super().sample_batch(batch_size)

        # Check if the indices are within bounds
        # check_indices(self.state, idx)
        # check_indices(self.next_state, idx)

        # Add the state and next_state to the batch
        batch["state"] = self.state[idx]
        batch["next_state"] = self.next_state[idx]
        
        return batch, idx
    
    
    def sample_batch_with_idx(self, idx):
        """
        Returns a batch of data samples with the given indices.

        Args:
            idx (list of int): The indices of the samples to be included in the batch.

        Returns:
            dict: A dictionary containing the batch of data samples. In particular, the dictionary contains: 
                - obs: The observations, at time t-1 and t;
                - next_obs: The next observations, at time t and t+1;
                - z_LF: The latent representation of the data samples (at time t-1 and t) at the low-fidelity level;
                - z_next_LF: The latent representation of the following data samples (at time t and t+1) at the low-fidelity level;
                - z_fwd_LF: The latent representation produced by the forward part of the model at the low-fidelity level;
                - state: The state at time t;
                - next_state: The state at time t+1.
        """
        
        batch = super().sample_batch_with_idx(idx)
        batch["state"] = self.state[idx]
        batch["next_state"] = self.next_state[idx]
        
        return batch


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
                - state: The state at time t;
                - next_state: The state at time t+1.
        """

        all_samples = super().get_all_samples()
        all_samples["state"] = self.state
        all_samples["next_state"] = self.next_state
        
        return all_samples
    

class PDEDataLoader(BaseDataLoader):
    """
    A class for loading and manipulating data for the multi-fidelity deep kernel learning model, using data from the PDE environment.
    It inherits from the BaseDataLoader class.

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
        t (torch.Tensor): The tensor containing the time steps of the data samples.
        done (torch.Tensor): The tensor indicating whether an episode is terminated at time t.

    Methods:
        sample_batch(batch_size=32): Returns a random batch of data samples.
        sample_batch_with_idx(idx): Returns a batch of data samples with the given indices.
        get_all_samples(): Returns all the samples and their low fidelity latent representations.
    """

    def __init__(self, data, z_LF, z_next_LF, z_fwd_LF, obs_dim):
        """
        Initializes the PDEDataLoader object.

        Args:
            data (list): A list of dictionaries containing the data.
            z_LF (float): The z value for low fidelity.
            z_next_LF (float): The z value for next low fidelity.
            z_fwd_LF (float): The z value for forward low fidelity.
            obs_dim (tuple): A tuple representing the dimensions of the observations.

        Returns:
            None
        """

        super().__init__(data, z_LF, z_next_LF, z_fwd_LF, obs_dim)
        self.t = torch.zeros(int(self.size), dtype=int)

        pos = 0
        for d in data:
            self.t[pos] = torch.tensor(d["t"])
            pos += 1


    def sample_batch(self, batch_size=32):
        """
        Returns a random batch of data samples.

        Args:
            batch_size (int, optional): The size of the batch. Defaults to 32.

        Returns:
            tuple:
                - batch (dict): A dictionary containing the batch of data samples. In particular, the dictionary contains: 
                    - obs: The observations, at time t-1 and t;
                    - next_obs: The next observations, at time t and t+1;
                    - z_LF: The latent representation of the data samples (at time t-1 and t) at the low-fidelity level;
                    - z_next_LF: The latent representation of the following data samples (at time t and t+1) at the low-fidelity level;
                    - z_fwd_LF: The latent representation produced by the forward part of the model at the low-fidelity level;
                    - t: The time steps of the data samples.
                - idx (torch.Tensor): The indices of the samples in the batch.
        """

        # Sample a batch
        batch, idx = super().sample_batch(batch_size)

        # Check if the indices are within bounds
        # check_indices(self.t, idx)

        # Add the time steps to the batch
        batch["t"] = self.t[idx]

        return batch, idx


    def sample_batch_with_idx(self, idx):
        """
        Returns a batch of data samples with the given indices.

        Args:
            idx (list of int): The indices of the samples to be included in the batch.

        Returns:
            dict: A dictionary containing the batch of data samples. In particular, the dictionary contains: 
                - obs: The observations, at time t-1 and t;
                - next_obs: The next observations, at time t and t+1;
                - z_LF: The latent representation of the data samples (at time t-1 and t) at the low-fidelity level;
                - z_next_LF: The latent representation of the following data samples (at time t and t+1) at the low-fidelity level;
                - z_fwd_LF: The latent representation produced by the forward part of the model at the low-fidelity level;
                - t: The time steps of the data samples.
        """

        batch = super().sample_batch_with_idx(idx)
        batch["t"] = self.t[idx]

        return batch
    

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
                - t: The time steps of the data samples.
        """

        all_samples = super().get_all_samples()
        all_samples["t"] = self.t

        return all_samples