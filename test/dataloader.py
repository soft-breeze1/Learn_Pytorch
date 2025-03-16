import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10(root='D:\Learn_Pytorch\dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
"""
drop_last=False
表示是否丢弃最后一个不完整的批次。当数据集中的样本数量不能被 batch_size 整除时，最后一个批次的样本数量会小于 batch_size。
如果 drop_last 设置为 True，则会丢弃这个不完整的批次；如果设置为 False，则会保留这个不完整的批次。
在测试阶段，通常建议将 drop_last 设置为 False，以确保使用所有的测试样本进行评估。
"""

# 测试数据集第一张图片及target
img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('D:\Learn_Pytorch\logs')
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step += 1

writer.close()

