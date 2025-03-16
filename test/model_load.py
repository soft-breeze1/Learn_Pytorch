import torch
import torchvision
from torch import nn
from model_save import *

# 【保存方式1 加载模型】
model = torch.load("D:\\Learn_Pytorch\\vgg16_method1.pth")
# print(model)

# 【保存方式2 加载模型】
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("D:\\Learn_Pytorch\\vgg16_method2.pth"))
# model = torch.load("D:\\Learn_Pytorch\\vgg16_method2.pth")     字典形式
# print(vgg16)

# 陷阱
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.model1 = Sequential(
#             Conv2d(3, 32, kernel_size=5,padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 32, kernel_size=5,padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 64, kernel_size=5,padding=2),
#             MaxPool2d(2),
#             Flatten(),
#             Linear(1024, 64),
#             Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model1(x)
#         return x

model = torch.load("D:\\Learn_Pytorch\\net_method1.pth")
print(model)

