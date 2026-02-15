import torch
import torch.nn as nn

# 1. The Generator (The AI Hacker)
# It takes random noise (Z) and tries to turn it into a realistic attack packet.
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh() # Forces output to be between -1 and 1
        )

    def forward(self, x):
        return self.net(x)

# 2. The Discriminator (The AI Police)
# It looks at traffic and decides: "Is this Real (1) or Fake/Generated (0)?"
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid() # Output probability
        )

    def forward(self, x):
        return self.net(x)