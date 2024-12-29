from src.cartpole.agent import Agent

class TestDQNAgent:
    def test_save_and_load_agent(self):
        data_dir = "test/data_dir/agent"
        expected_agent = Agent(4, 2, data_dir=data_dir)
        expected_agent.save_agent()
        
        actual_agent = Agent(4, 2, restore=True, data_dir=data_dir)
        actual_agent.load_agent()
        
        assert str(expected_agent.policy_net.state_dict()) == str(actual_agent.policy_net.state_dict())
        assert str(expected_agent.target_net.state_dict()) == str(actual_agent.target_net.state_dict())
