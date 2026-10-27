import random
from collections import deque


class ReplayMemory:
    """Stores (state, action, new_state, reward, terminated) and hands back random batches."""

    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)  # oldest experiences drop off when full

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
