import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='D:\Learn_Pytorch\dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

input = torch.tensor([[1,-0.5],
                      [-1,3]])

input = torch.reshape(input, (-1,1,2,2))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

net = Net()

writer = SummaryWriter('../logs_Sigmoid')
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images('input',imgs,step)
    outputs = net(imgs)
    writer.add_images('outputs',outputs,step)
    step += 1

writer.close()