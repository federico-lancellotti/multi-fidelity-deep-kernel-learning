import numpy as np


class DataLoader:
    def __init__(self, data, z_LF, obs_dim):
        self.size = len(data)
        self.obs = np.zeros(
            [int(self.size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])],
            dtype=np.float32,
        )
        self.z_LF = z_LF
        self.done = np.zeros(int(self.size), dtype=bool)

        pos = 0
        for d in data:
            self.obs[pos] = d[0].astype(np.float32) / 255
            self.done[pos] = d[1]
            pos = pos + 1


    # Returns a (list of) np.array random batches of dimension batch_size.
    # Each element of the list is a batch at the respective level of fidelity.
    def sample_batch(self, batch_size=32):
        idx = np.random.randint(0, self.size, batch_size)

        for i in range(len(idx)):
            if self.done[i] == True:
                idx[i] = idx[i] - 1

        batch = np.array([self.obs[i] for i in idx])
        batch_z_LF = np.array([self.z_LF[i] for i in idx])

        return batch, batch_z_LF

    # Returns a np.array with all the samples
    def get_all_samples(self):
        return np.array(self.obs), self.z_LF
