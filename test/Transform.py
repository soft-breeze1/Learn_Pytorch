from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法 -》 tensor数据类型
# 通过 transforms.ToTensor去看两个问题
# 1、transforms如何使用（python）
# 2、为什么我们需要Tensor数据类型

img_path = "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)                                   # 输入

writer = SummaryWriter("logs")


tensor_trans = transforms.ToTensor()                         # 输出
# 调用 ToTensor 实例的 __call__ 方法，将输入的图像 img 转换为 PyTorch 张量。
tensor_img = tensor_trans(img)


writer.add_image("Tensor_img",tensor_img)          # 将图像数据写入 TensorBoard 日志

writer.close()
