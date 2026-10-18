"""Uniform transition replay for scalar-reward SAC; no ensemble membership."""

import copy

import numpy as np
import torch


class Replay:
    def __init__(self, capacity, observation_dim, seed):
        if (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0
                or isinstance(observation_dim, bool) or not isinstance(observation_dim, int)
                or observation_dim <= 0):
            raise ValueError("Replay capacity and observation_dim must be positive integers")
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.rng = np.random.default_rng(seed)
        self.cursor = self.size = 0
        self.data = {
            key: np.zeros((capacity, width), np.float32)
            for key, width in (("obs", observation_dim), ("action", 2), ("reward", 1),
                               ("next_obs", observation_dim), ("done", 1))
        }

    def __len__(self):
        return self.size

    def add(self, obs, raw_action, reward, next_obs, done):
        values = (obs, raw_action, [reward], next_obs, [done])
        converted = {}
        for (key, array), value in zip(self.data.items(), values):
            value = np.asarray(value, dtype=np.float32)
            if value.shape != array.shape[1:] or not np.isfinite(value).all():
                raise ValueError(f"Invalid finite replay field: {key}")
            converted[key] = value
        if np.any(np.abs(converted["action"]) > 1):
            raise ValueError("Replay actions must be raw controls in [-1, 1]")
        if converted["done"][0] not in (0, 1):
            raise ValueError("Replay done must be 0 or 1")
        for key, value in converted.items():
            self.data[key][self.cursor] = value
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size, device):
        if (isinstance(batch_size, bool) or not isinstance(batch_size, int)
                or batch_size <= 0 or self.size < batch_size):
            raise ValueError("Replay needs at least batch_size transitions")
        indices = self.rng.integers(self.size, size=batch_size)
        return {key: torch.as_tensor(array[indices], device=device)
                for key, array in self.data.items()}

    def state_dict(self):
        return {"capacity": self.capacity, "observation_dim": self.observation_dim,
                "cursor": self.cursor, "size": self.size,
                "data": {key: array[:self.size].copy() for key, array in self.data.items()},
                "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if (state["capacity"] != self.capacity
                or state["observation_dim"] != self.observation_dim):
            raise ValueError("Replay checkpoint dimensions differ")
        size, cursor = state["size"], state["cursor"]
        if (not isinstance(size, int) or not isinstance(cursor, int)
                or not 0 <= size <= self.capacity or not 0 <= cursor < self.capacity
                or (size < self.capacity and cursor != size)):
            raise ValueError("Invalid replay checkpoint position")
        if state["data"].keys() != self.data.keys():
            raise ValueError("Invalid replay checkpoint fields")
        for key, array in self.data.items():
            values = np.asarray(state["data"][key])
            if values.shape != (size, array.shape[1]) or not np.isfinite(values).all():
                raise ValueError(f"Invalid replay checkpoint field: {key}")
        rng = np.random.default_rng(0)
        rng.bit_generator.state = copy.deepcopy(state["rng"])
        for key, array in self.data.items():
            array[:size] = state["data"][key]
        self.size, self.cursor, self.rng = size, cursor, rng
