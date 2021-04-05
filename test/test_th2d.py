import unittest
import src.envs.TreasureHunt2D as th2d


class TestTH2D(unittest.TestCase):
    def setUp(self):
        self.env = th2d.TreasureHunt2D()

    def test_action_filter(self):
        self.env.reset()
        actions = [(0, 1), (1, 0), (1, 0), (0, -1), (-1, 0), (-1, 0)]
        f_actions_assertion = [
            [(0, 1), (1, 0), (0, -1)],
            [(0, 1), (1, 0), (0, -1), (-1, 0)],
            [(0, 1), (1, 0), (0, -1), (-1, 0)],
            [(0, 1), (1, 0), (-1, 0)],
            [(0, 1), (1, 0), (-1, 0)],
            [(0, 1), (1, 0)],
        ]

        for action, f_ass in zip(actions, f_actions_assertion):
            self.env.step(action)
            f_actions = self.env.action_filter(self.env.observation)
            self.assertCountEqual(f_actions, f_ass)

        self.env.reset()
        actions = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        for action in actions: 
            self.env.step(action)

        f_actions = self.env.action_filter(self.env.observation)
        f_ass = [(1, 0), (0, -1)]
        self.assertCountEqual(f_actions, f_ass)

