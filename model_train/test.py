import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


image_path = "D:\\Learn_Pytorch\\Image_test\\airplane.png"
image = Image.open(image_path)
print(image)

image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5,stride=1,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5,stride=1,padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5,stride=1,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

model = Net()

model.load_state_dict(torch.load("D:\\Learn_Pytorch\\model_train\\net_29_gpu.pth"))
print(model)
image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(dim=1))
