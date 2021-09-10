import torch
from torch import nn


class Gnet(nn.Module):
    def __init__(self):
        super(Gnet, self).__init__()
        self.Con1_layer = nn.ConvTranspose2d(64, 256, 3, bias=False)
        self.Con2_layer = nn.ConvTranspose2d(10, 256, 3, bias=False)
        self.ConT_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 3, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 3, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, 3, 2, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input, lable):
        lable = lable.reshape(-1, 10, 1, 1)
        out1 = self.Con1_layer(input)
        out2 = self.Con2_layer(lable)
        out  = torch.cat([out1, out2], dim=1)
        out  = self.ConT_layer(out)
        return out

class Dnet(nn.Module):
    def __init__(self):
        super(Dnet, self).__init__()
        self.lin = nn.Linear(10, 28*28)
        self.Con = nn.Sequential(
            nn.Conv2d(2, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, 3, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input, lable):
        # lable = lable.reshape(-1, 10)
        input = input.reshape(-1, 1, 28, 28)
        lable = self.lin(lable)
        lable = lable.reshape(-1, 1, 28, 28)
        out   = torch.cat([input, lable], dim=1)
        out   = self.Con(out)
        out   = out.reshape(-1)
        return out
