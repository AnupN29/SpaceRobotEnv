import numpy as np
from RL_algorithms.Torch.SAC.SAC_ENV import core
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        
        self.img_obs_buf = np.zeros(core.combined_shape(size, (3,obs_dim,obs_dim)), dtype=np.float32)
        self.depth_obs_buf = np.zeros(core.combined_shape(size, (1, obs_dim,obs_dim)), dtype=np.float32)
        
        self.expert_act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, expert_act):
        self.img_obs_buf[self.ptr] = obs[0].cpu()
        self.depth_obs_buf[self.ptr] = obs[1].cpu()

        self.expert_act_buf[self.ptr] = expert_act

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, device=device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(img_obs=self.img_obs_buf[idxs],
                     depth_obs=self.depth_obs_buf[idxs],
                     expert_act=self.expert_act_buf[idxs])
        
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k,v in batch.items()}