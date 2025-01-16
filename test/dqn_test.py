from src.cartpole.dqn import DQN
import os
import torch

class TestDQN:
    def test_save_and_load_model(self):
        data_dir = "test/data_dir/dqn"
        os.makedirs(data_dir, exist_ok=True)

        expected_dqn = DQN("test", 4, 2, data_dir)
        expected_dqn.save_model()
        
        actual_dqn = DQN("test", 4, 2, data_dir)
        assert str(expected_dqn.state_dict()) != str(actual_dqn.state_dict())
        
        actual_dqn.load_model()
        assert str(expected_dqn.state_dict()) == str(actual_dqn.state_dict())
        
    def test_input_output_shapes(self):
        dqn = DQN("test", 4, 2)
        
        input_shape = (4, 4)
        input = torch.randn(input_shape)
        output = dqn(input)
        
        expected_shape = (4, 2)
        assert expected_shape == output.shape
