import unittest
import src.envs.TreasureHunt2D as th2d


class TestTH2D(unittest.TestCase):
    def setUp(self):
        self.env = th2d.TreasureHunt2D()

    def test_action_filter(self):
        self.env.reset()
        actions = [(0, 1)]#, (1, 0), (1, 0), (0, -1), (0, -1)]
        f_actions_assertion = [
            [(0, 1), (1, 0), (0, -1)],
        ]
        for action, f_ass in zip(actions, f_actions_assertion):
            f_actions = self.env.action_filter(self.env.observation)
            self.assertEqual(f_actions, f_ass)
            self.env.step(action)



