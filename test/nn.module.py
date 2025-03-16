import torch
from torch import nn


class AoNet(nn.Module):
    def __init__(self):
        super(AoNet,self).__init__()

    def forward(self,input):
        output=input+1
        return output

AoNet = AoNet()
x = torch.tensor(1.0)
output = AoNet(x)
print(output)
