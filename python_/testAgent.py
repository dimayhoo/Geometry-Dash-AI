import unittest
import torch as th
from agent import Agent

''' NOTE: why outputs repeat?
They repeat because each test re-initializes the model or environment and prints the same message. You can fix it by removing duplicated print/log calls or reusing the same Agent across tests. For example, only initialize the model in setUpClass or a fixture so it doesn’t run multiple times per test.'''

class TestAgentIndependent(unittest.TestCase):
    def setUp(self):
        # Create minimal model parameters and agent
        model_params = {"name": "PPO", "to_init": True}
        self.test_agent = Agent(model_params=model_params, lvl_id=1)

    def test_agent_initialization(self):
        self.assertIsNotNone(self.test_agent.env, "Agent environment should be initialized.")
        self.assertIsNotNone(self.test_agent.model, "Agent model should be initialized.")

    def test_agent_state_retrieval(self):
        coli = 0
        state = self.test_agent.get_state(coli)
        # State is typically a tuple or dict. Just check it's not None.
        self.assertIsNotNone(state, "Agent should retrieve a valid state.")

    def test_agent_action_random(self):
        coli = 0
        state = self.test_agent.get_state(coli)
        # For a random action
        action = self.test_agent.get_action(state, random=True)
        self.assertIn(action, [0, 1], "Action should be a valid discrete action.")

    def test_agent_action_state(self):
        coli = 12
        state = self.test_agent.get_state(coli)
        # For a random action
        action = self.test_agent.get_action(state)
        self.assertIn(action, [0, 1], "Action should be a valid discrete action.")
    
    def test_agent_state_multiple_indices(self):
        for coli in [5, 10, 15]:
            state = self.test_agent.get_state(coli)
            self.assertIsNotNone(state, f"State should be valid for coli={coli}")
            lvl_frame, other_params = state["lvl_frame"], state["other_params"]
            print(f"coli={coli}, lvl_frame shape={lvl_frame[0].shape if lvl_frame is not None else None}")

    def test_agent_state_edge_case(self):
        # Provide an edge column index (e.g., near end of level)
        coli = self.test_agent.lvl_matrix.size(1) - 1
        state = self.test_agent.get_state(coli)
        self.assertIsNotNone(state, "State should be valid even at the last column.")
        print(f"Edge coli={coli}, state={state}")

    def test_agent_actions_matrix(self):
        # Provide a state with a matrix
        A = self.test_agent.get_actions_matrix()
        self.assertIsNotNone(A, "Actions matrix should be valid.")
        print(f"Actions matrix: {A}")

if __name__ == "__main__":
    unittest.main()