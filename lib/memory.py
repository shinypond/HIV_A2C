import numpy as np
import torch


class VanillaReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = torch.tensor([]) # shape : N x (6+1+1+6)
        self.position = 0
        self.zero = torch.tensor([[0.] * 14])
        
    def push(self, state, action_idx, reward, next_state):
        if self.memory.shape[0] == 0:
            self.memory = self.zero
        elif self.memory.shape[0] < self.capacity:
            self.memory = torch.cat([self.memory, self.zero], dim=0)
        self.memory[self.position, :] = torch.cat(
            [state, action_idx.unsqueeze(0), reward.unsqueeze(0), next_state], dim=0
        )
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return self.memory[np.random.choice(self.memory.shape[0], batch_size), :]
    
    def __len__(self):
        return self.memory.shape[0]
        

class PrioritizedReplayMemory:
    e = 0.01
    a = 1.0 # 0.6
    beta = 0.4
    # beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.capacity = capacity
        self.priorities = np.array([], dtype=np.float32)
        self.data = None

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.priorities = np.append(self.priorities, p)
        if self.data is None:
            self.data = sample.clone()
        else:
            self.data = torch.cat([self.data.cpu(), sample.clone()], dim=0)

    def sample(self, n):
        data_idxs = np.arange(self.data.shape[0])
        p = self.priorities / self.priorities.sum()
        
        batch_idxs = np.random.choice(data_idxs, n, p=p)

        batch = self.data[batch_idxs]

        batch_priorities = self.priorities[batch_idxs]
        batch_priorities /= batch_priorities.sum()
        is_weight = np.power(self.data.shape[0] * batch_priorities, -self.beta)
        is_weight /= is_weight.max()

        return batch, batch_idxs, is_weight

    def update(self, idxs, errors):
        p = self._get_priority(errors)
        self.priorities[idxs] = p.reshape(-1)
