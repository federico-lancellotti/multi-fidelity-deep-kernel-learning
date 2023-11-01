import numpy as np


class DataLoader:
    def __init__(self, data, obs_dim):
        self.size = len(data)
        self.obs = np.zeros(
            [int(self.size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])],
            dtype=np.float32,
        )
        self.done = np.zeros(int(self.size), dtype=bool)

        pos = 0
        for d in data:
            self.obs[pos] = d[0].astype(np.float32) / 255
            self.done[pos] = d[1]
            pos = pos + 1

    # Returns a np.array random batch of dimension batch_size
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, len(self.obs), batch_size)

        for i in range(len(idxs)):
            if self.done[i] == True:
                idxs[i] = idxs[i] - 1

        batch = [self.obs[i] for i in idxs]

        return np.array(batch)
