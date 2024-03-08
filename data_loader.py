import numpy as np


class DataLoader:
    def __init__(self, data, z_LF, z_next_LF, z_fwd_LF, obs_dim):
        self.size = len(data)
        self.obs = np.zeros([int(self.size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])], dtype=np.float32)
        self.next_obs = np.zeros([int(self.size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])], dtype=np.float32)
        self.z_LF = z_LF
        self.z_next_LF = z_next_LF
        self.z_fwd_LF = z_fwd_LF
        self.done = np.zeros(int(self.size), dtype=bool)

        pos = 0
        for d in data:
            self.obs[pos] = d["obs"].astype(np.float32) / 255
            self.next_obs[pos] = d["next_obs"].astype(np.float32) / 255
            self.done[pos] = d["terminated"]
            pos = pos + 1

    # Returns a (list of) np.array random batches of dimension batch_size.
    # Each element of the list is a batch at the respective level of fidelity.
    def sample_batch(self, batch_size=32):
        idx = np.random.randint(0, self.size, batch_size)

        for i in range(len(idx)):
            if self.done[i] == True:
                idx[i] = idx[i] - 1

        #batch = np.array([self.obs[i] for i in idx])
        #batch_z_LF = np.array([self.z_LF[i] for i in idx])

        return dict(obs = self.obs[idx], 
                    next_obs = self.next_obs[idx],
                    z_LF = self.z_LF[idx],
                    z_next_LF = self.z_next_LF[idx],
                    z_fwd_LF = self.z_fwd_LF[idx])

    # Returns a np.array with all the samples and the latent representation 
    # of the l-1 fidelity level.
    def get_all_samples(self):
        #return np.array(self.obs), self.z_LF
        return dict(obs = self.obs, 
                    next_obs = self.next_obs,
                    z_LF = self.z_LF,
                    z_next_LF = self.z_next_LF,
                    z_fwd_LF = self.z_fwd_LF)
