import unittest
import src.AI.agent as agent
import src.envs.TreasureHunt as th

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.env = th.TreasureHunt()

    def test_argmax(self):
        ag = agent.Agent(env=self.env, initial_q_mode="zero")
        state = 1
        action = 1
        ag.q_table.at[state, action] = 100
        m_action = ag.argmax(ag.q_table, state)

        self.assertEqual(action, m_action)
        
        actions = [1, -1]
        m_action = ag.argmax(ag.q_table, state, actions)
        self.assertEqual(action, m_action)
        
