import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
dataset = torchvision.datasets.CIFAR10(root='D:\Learn_Pytorch\dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64)

# input =torch.tensor([[1,2,0,3,1],
#                      [0,1,2,3,1],
#                      [1,2,1,0,0],
#                      [5,2,3,1,1],
#                      [2,1,0,1,1]])
#
# input = torch.reshape(input,(-1,1,5,5))
# print(input.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

Net = Net()
# output = Net(input)
# print(output)

writer = SummaryWriter('../logs_maxpool')
step = 0
for data in dataloader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("maxpoolinput", imgs, step)
    output = Net(imgs)
    writer.add_images("maxpooloutput", output, step)
    step += 1

writer.close()