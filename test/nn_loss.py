import torch
from numpy import dtype
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1,2,3],dtype = torch.float32)
targets = torch.tensor([1,2,5],dtype = torch.float32)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

loss = L1Loss()
result = loss(inputs,targets)

loss_Mae = MSELoss()
result_Mse = loss_Mae(inputs,targets)

print(result)
print(result_Mse)


# 交叉熵
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x,y)
print(result_cross)