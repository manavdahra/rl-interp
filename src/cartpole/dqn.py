import torch
import torch.nn as nn
import os
import shutil
from collections import OrderedDict

class DQN(nn.Module):
    def __init__(self, name, n_observations, n_actions, data_dir="."):
        super(DQN, self).__init__()
        self.name = name
        self.filepath = f"{data_dir}/{self.name}.ckpt"
        n_hidden = 128
        self.network = nn.Sequential(OrderedDict(
            [
                ("input", nn.Linear(n_observations, n_hidden)),
                ("act_input", nn.ReLU()),
                ("hidden", nn.Linear(n_hidden, n_hidden)),
                ("act_hidden", nn.ReLU()),
                ("output", nn.Linear(n_hidden, n_actions)),
            ])
        )
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
        
        self.network.apply(init_weights)
        
        self.activations = {}
        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        self.network.input.register_forward_hook(getActivation("input"))
        self.network.act_input.register_forward_hook(getActivation("act_input"))
        self.network.hidden.register_forward_hook(getActivation("hidden"))
        self.network.act_hidden.register_forward_hook(getActivation("act_hidden"))
        self.network.output.register_forward_hook(getActivation("output"))
    
    def _reset_disk_state(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        open(self.filepath, "w").close()

    def forward(self, x):
        return self.network(x)
    
    def get_activations(self) -> dict:
        return self.activations

    def save_model(self):
        print(f"Saving the model at: {self.filepath}")
        self._reset_disk_state()
        
        torch.save(self.state_dict(), self.filepath)
    
    def load_model(self):
        print(f"Loading model parameters from: {self.filepath}")
        if not os.path.exists(self.filepath):
            print(f"could not find data at {self.filepath}")
            return
        
        state = torch.load(self.filepath, weights_only=True)
        self.load_state_dict(state)
