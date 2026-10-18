import unittest

import numpy as np

from eqnsac.replay import Replay


class ReplaySnapshotTests(unittest.TestCase):
    @staticmethod
    def add(replay, value):
        replay.add(np.full(19, value), [0, 0], [0, 0], np.zeros(19), False)

    def test_snapshot_is_independent_of_future_insertions(self):
        replay = Replay(4, 2, 1, 1)
        self.add(replay, 10)
        snapshot = replay.state_dict()
        self.add(replay, 20)
        self.assertEqual(snapshot["pools"], [[0], [0]])
        restored = Replay(4, 2, 1, 2)
        restored.load_state_dict(snapshot)
        for member in range(2):
            np.testing.assert_array_equal(
                restored.sample(8, "cpu", member)["state"], np.full((8, 19), 10)
            )

    def test_restored_replays_do_not_share_membership_lists(self):
        source = Replay(4, 2, 1, 1)
        self.add(source, 10)
        snapshot = source.state_dict()
        first = Replay(4, 2, 1, 2)
        second = Replay(4, 2, 1, 3)
        first.load_state_dict(snapshot)
        second.load_state_dict(snapshot)
        self.add(first, 20)
        self.assertEqual(snapshot["pools"], [[0], [0]])
        self.assertEqual(second.pools, [[0], [0]])
        for member, pool in enumerate(second.pools):
            np.testing.assert_array_equal(second.positions[member, pool], np.arange(len(pool)))


if __name__ == "__main__":
    unittest.main()
