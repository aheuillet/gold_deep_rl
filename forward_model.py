import torch
import torch.nn as nn
# based on pytorch tutorial by yfeng997: https://github.com/yfeng997/MadMario/blob/master/neural.py

class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super().__init__()
    
        self.fc = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        return self.fc(torch.cat([state, action]))
