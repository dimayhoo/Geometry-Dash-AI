import unittest
from agent import Agent
from levelStructure import get_addition_i
import torch as th
import pathlib

''' NOTE: why outputs repeat?
They repeat because each test re-initializes the model or environment and prints the same message. You can fix it by removing duplicated print/log calls or reusing the same Agent across tests. For example, only initialize the model in setUpClass or a fixture so it doesn’t run multiple times per test.'''

class TestAgentIndependent(unittest.TestCase):
    def setUp(self):
        lvl_id = 1

        # Create minimal model parameters and agent
        model_params = {"name": "PPO", "to_init": True}
        self.test_agent = Agent(model_params=model_params, lvl_id=lvl_id)
        self.lvl_id = lvl_id

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
            #print(state)
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
        #print(f"Actions matrix: {A}")
    
    '''def test_agent_level(self):
        for i in range(61, 131, 10):
            print(self.test_agent.lvl_matrix[:, i - 10:i])
        lvl_yPos = self.test_agent.lvl_matrix[get_addition_i("yPos")]
        print(lvl_yPos[:100])
        print(f"Min yPos is {lvl_yPos.min()}, max yPos is {lvl_yPos.max()}")'''

        # NOTE: there are some errorneous values in the space with ids -2, 7. 

    '''def test_get_state(self):
        for coli in range(11, 121, 10):
            print(self.test_agent.get_state(coli))'''

    def test_lvl_matrix(self, postfix=""):
        print(self.test_agent.lvl_matrix)
        
        # Create directory if it doesn't exist
        dir_path = pathlib.Path(__file__).parent / "levels"
        dir_path.mkdir(exist_ok=True)
        
        # Save in PyTorch format
        pt_path = dir_path / f"{self.lvl_id}{postfix}.pt"
        th.save(self.test_agent.lvl_matrix, pt_path)
        
        # Save as CSV for analysis
        import numpy as np
        csv_path = dir_path / f"{self.lvl_id}{postfix}.csv"
        np_array = self.test_agent.lvl_matrix.cpu().numpy()
        np.savetxt(csv_path, np_array, delimiter=',', fmt='%.1f')
        
        # Save as text with complete matrix (no truncation)
        txt_path = dir_path / f"{self.lvl_id}{postfix}_readable.txt"
        with open(txt_path, "w") as f:
            f.write(f"Level {self.lvl_id} matrix shape: {self.test_agent.lvl_matrix.shape}\n\n")
            # Set numpy options to prevent truncation
            np.set_printoptions(threshold=np.inf, linewidth=200)
            f.write(str(np_array))
            # Reset numpy print options to defaults
            np.set_printoptions(threshold=1000, linewidth=75)
        
        print(f"Level matrix saved to multiple formats for analysis")

        ''' 
        9.0 and -25 are blokcs too, where -25 is ending one.
        29 and 30, maybe, to tag as 0?
        
        - Level matrix isn't modified anywhere in the prog except for additional params.
        '''

        


if __name__ == "__main__":
    unittest.main()