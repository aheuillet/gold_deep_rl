import torch.nn as nn
import copy
# based on pytorch tutorial by yfeng997: https://github.com/yfeng997/MadMario/blob/master/neural.py

class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        f, c, h, w = input_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=f*c, out_channels=32, kernel_size=6, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(14336,512),
            nn.ELU(),
        )
    
        self.fc = nn.Linear(512, output_dim)
        self.online = nn.Sequential(self.cnn, self.fc)

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
           p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
