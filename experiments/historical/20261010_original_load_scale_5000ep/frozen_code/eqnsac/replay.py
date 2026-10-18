"""Bootstrap membership fixed on insertion; O(1) pool maintenance on overwrite."""

import numpy as np
import torch


class Replay:
    def __init__(self, capacity, members, probability, seed):
        self.capacity = capacity
        self.members = members
        self.probability = probability
        self.rng = np.random.default_rng(seed)
        self.cursor = 0
        self.size = 0
        self.data = {
            k: np.zeros((capacity, d), np.float32)
            for k, d in [
                ("state", 19),
                ("action", 2),
                ("reward", 2),
                ("next_state", 19),
                ("done", 1),
            ]
        }
        self.pools = [[] for _ in range(members)]
        self.positions = np.full((members, capacity), -1, dtype=np.int64)

    def add(self, state, action, reward, next_state, done):
        idx = self.cursor
        for key, value in zip(self.data, (state, action, reward, next_state, done)):
            self.data[key][idx] = value
        mask = self.rng.random(self.members) < self.probability
        for member, pool in enumerate(self.pools):
            position = self.positions[member, idx]
            if position >= 0:
                last = pool.pop()
                if position < len(pool):
                    pool[position] = last
                    self.positions[member, last] = position
                self.positions[member, idx] = -1
            if mask[member]:
                self.positions[member, idx] = len(pool)
                pool.append(idx)
        self.cursor = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def ready(self, batch_size):
        return all(len(p) >= batch_size for p in self.pools)

    def sample(self, batch_size, device, member=None):
        if member is None:
            indices = self.rng.integers(self.size, size=batch_size)
        else:
            pool = self.pools[member]
            indices = np.asarray(
                [pool[i] for i in self.rng.integers(len(pool), size=batch_size)]
            )
        return {
            k: torch.as_tensor(v[indices], device=device) for k, v in self.data.items()
        }

    def state_dict(self):
        return dict(
            data={k: v[: self.size].copy() for k, v in self.data.items()},
            pools=self.pools,
            cursor=self.cursor,
            size=self.size,
            rng=self.rng.bit_generator.state,
        )

    def load_state_dict(self, state):
        self.cursor = state["cursor"]
        self.size = state["size"]
        for key, value in state["data"].items():
            self.data[key][: self.size] = value
        self.pools = state["pools"]
        self.positions.fill(-1)
        for member, pool in enumerate(self.pools):
            self.positions[member, pool] = np.arange(len(pool))
        self.rng.bit_generator.state = state["rng"]
