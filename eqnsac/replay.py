"""Bootstrap membership fixed on insertion; O(1) pool maintenance on overwrite."""

import numpy as np
import torch

from .environment import ACTION_DIM, STATE_DIM


class Replay:
    """每条转移仅存一次，各成员用索引池表示自己的bootstrap子集。

    data: state/next_state [capacity,19]，执行action [capacity,2]，
    reward [capacity,2]（价格、储能目标），done [capacity,1]。
    positions[member, slot]为slot在成员池中的位置，-1表示不属于该池。
    """

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
                ("state", STATE_DIM),
                ("action", ACTION_DIM),
                ("reward", 2),
                ("next_state", STATE_DIM),
                ("done", 1),
            ]
        }
        self.pools = [[] for _ in range(members)]
        self.positions = np.full((members, capacity), -1, dtype=np.int64)

    def add(self, state, action, reward, next_state, done):
        """插入时一次性决定成员归属；回放采样时不重新抽bootstrap mask。"""
        idx = self.cursor
        for key, value in zip(self.data, (state, action, reward, next_state, done)):
            self.data[key][idx] = value
        mask = self.rng.random(self.members) < self.probability
        for member, pool in enumerate(self.pools):
            position = self.positions[member, idx]
            if position >= 0:
                # Swap with the last entry to remove an overwritten slot in O(1).
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
        """member=None从全回放采Actor状态，否则从指定Critic成员池采样。"""
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
            pools=[pool.copy() for pool in self.pools],
            cursor=self.cursor,
            size=self.size,
            rng=self.rng.bit_generator.state,
        )

    def load_state_dict(self, state):
        self.cursor = state["cursor"]
        self.size = state["size"]
        for key, value in state["data"].items():
            self.data[key][: self.size] = value
        self.pools = [pool.copy() for pool in state["pools"]]
        self.positions.fill(-1)
        for member, pool in enumerate(self.pools):
            self.positions[member, pool] = np.arange(len(pool))
        self.rng.bit_generator.state = state["rng"]
