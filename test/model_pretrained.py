import torch
import torchvision
from torch import nn
from torchvision.models.vgg import VGG16_Weights

# train_data = torchvision.datasets.ImageNet("../data_image_net", train=True, download=True, transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(weights=None)
print(vgg16_false)

# 加载预训练模型
vgg16_true = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', weights=None)
vgg16_true.load_state_dict(torch.load('D:\\Learn_Pytorch\\vgg16-397923af.pth'))
# print(vgg16_true)
# print(ok)

train_data = torchvision.datasets.CIFAR10(root='D:\Learn_Pytorch\dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

# 迁移学习，在VGG16的最后一层的输出1000个类别改为10分类，加一个Linear层
vgg16_true.classifier. add_module('add_linear',nn.Linear(1000, 10))
print(vgg16_true)

# 将 classifier 中的第六层的输入4096，输出1000   -》    输入4096，输出10
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)


