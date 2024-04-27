import numpy as np
import torch


class DataLoader:
    def __init__(self, data, z_LF, z_next_LF, z_fwd_LF, obs_dim):
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

    # Returns a (list of) np.array random batches of dimension batch_size.
    # Each element of the list is a batch at the respective level of fidelity.
    def sample_batch(self, batch_size=32):
        idx = torch.randint(0, self.size, (batch_size,))

        for i in range(len(idx)):
            if self.done[i] == True:
                idx[i] = idx[i] - 1

        return dict(obs = self.obs[idx], 
                    next_obs = self.next_obs[idx],
                    z_LF = self.z_LF[idx],
                    z_next_LF = self.z_next_LF[idx],
                    z_fwd_LF = self.z_fwd_LF[idx],
                    state = self.state[idx],
                    next_state = self.next_state[idx])

    # Returns a np.array with all the samples and the latent representation 
    # of the l-1 fidelity level.
    def get_all_samples(self):
        return dict(obs = self.obs, 
                    next_obs = self.next_obs,
                    z_LF = self.z_LF,
                    z_next_LF = self.z_next_LF,
                    z_fwd_LF = self.z_fwd_LF,
                    state = self.state,
                    next_state = self.next_state)