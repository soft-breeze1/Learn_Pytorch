from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
# 从文件系统中读取一张图像，并将其转换为 NumPy 数组
image_path = "data/val/ants/17081114_79b9a27724.jpg"
img_PIL = Image.open(image_path)
img_array= np.array(img_PIL)                 # 将 img_PIL 转换为 NumPy 数组
print(type(img_array))
print(img_array.shape)

writer.add_image("train", img_array,1,dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()