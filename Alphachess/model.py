import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 3 bloques residuales (ligero para CPU)
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.res3 = ResBlock(64)
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, 1),
            nn.Flatten(),
            nn.Linear(2*8*8, 4096),
            nn.LogSoftmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Flatten(),
            nn.Linear(8*8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.res1(x); x = self.res2(x); x = self.res3(x)
        return self.policy_head(x), self.value_head(x)

class ResBlock(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.c1 = nn.Conv2d(n, n, 3, padding=1)
        self.b1 = nn.BatchNorm2d(n)
        self.c2 = nn.Conv2d(n, n, 3, padding=1)
        self.b2 = nn.BatchNorm2d(n)
    def forward(self, x):
        res = x
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return F.relu(x + res)