import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='D:\Learn_Pytorch\dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = Conv2d(3, 32, kernel_size=5,padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, kernel_size=5,padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, kernel_size=5,padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024,64)
        # self.linear2 = Linear(64,10)

        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5,padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x

net = Net()

# 损失函数
loss = nn.CrossEntropyLoss()

for data in dataloader:
    imgs,targets = data
    output = net(imgs)
    # print(targets)
    # print(output)
    result_loss = loss(output, targets)
    # 反向传播
    result_loss.backward()