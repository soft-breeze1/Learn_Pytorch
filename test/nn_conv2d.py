import torch
import torchvision
from torch import nn
from torch import conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='D:\Learn_Pytorch\dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

Net = Net()
print(Net)

writer = SummaryWriter('D:\Learn_Pytorch\logs')
step = 0
for data in dataloader:
    imgs,targets = data
    output = Net(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input",imgs,step)
    # torch.size([64，6，30，30])->[xxx，3，30，30]
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("ouput", output, step)
    step += 1

writer.close()