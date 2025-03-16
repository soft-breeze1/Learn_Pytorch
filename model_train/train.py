import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.utils.data import DataLoader
import time


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='D:\Learn_Pytorch\dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_data = torchvision.datasets.CIFAR10(root='D:\Learn_Pytorch\dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

# 利用DataLoader来加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)


# 创建网络模型
net = Net()


# 创建损失函数
loss_fn = nn.CrossEntropyLoss()


# 优化器
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate)


# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10


# 添加 tensorboard
writer = SummaryWriter(log_dir='./logs_train')


start_time = time.time()
for i in range(epoch):
    print(f"--------第{i+1}轮训练开始--------")

    # 训练步骤开始
    net.train()
    for data in train_dataloader:
        imgs, labels = data
        outputs = net(imgs)
        loss = loss_fn(outputs, labels)          # 使用损失函数 loss_fn 计算模型输出 outputs 与真实标签 labels 之间的损失值 loss。

        # 优化器优化模型
        optimizer.zero_grad()                    # 在每次反向传播之前，将优化器中所有参数的梯度清零，避免梯度累积。
        loss.backward()                          # 进行反向传播，计算损失函数关于模型参数的梯度
        optimizer.step()                         # 根据计算得到的梯度，使用优化器 optimizer 更新模型的参数。

        total_train_step += 1                    # 记录总的训练步数，每完成一个批次的训练，训练步数加 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(f"训练次数:{total_train_step},Loss：{loss.item()}")
            writer.add_scalar('train_loss',loss.item(),total_train_step)

    # 测试步骤开始
    net.eval()
    total_test_loss = 0
    total_accuracy = 0            # 整体正确率
    with torch.no_grad():
        for data in test_dataloader:
            imgs, labels = data
            outputs = net(imgs)
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()
            # 分类正确率
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy = total_accuracy + accuracy

    print(f"整体测试集上的Loss:{total_test_loss}")
    print(f"整体测试集上的正确率Accuracy:{total_accuracy/train_data_size}")
    writer.add_scalar('test_loss',total_test_loss,total_test_step)
    writer.add_scalar('test_accuracy',total_accuracy,total_test_step)
    total_test_step = total_test_step + 1

    torch.save(net,r'D:\Learn_Pytorch\model_train\net_train_epoch\net_{}.pth'.format(i))
    # torch.save(net.state_dict(),r'D:\Learn_Pytorch\model_train\net_train_epoch\net_{}.pth'.format(i))
    print("模型已保存")

writer.close()
