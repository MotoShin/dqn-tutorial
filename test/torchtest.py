import torch
import torch.nn as nn

m1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
m2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
m3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
fc4 = nn.Linear(7 * 7 * 32, 200)
fc5 = nn.Linear(200, 200)
mu = nn.Linear(200, 2)
input = torch.randn(1, 4, 84, 84)
output = m1(input)
print(output.shape)
output = m2(output)
print(output.shape)
output = m3(output)
print(output.shape)
print(output.view(output.size(0), -1).shape)
output = fc4(output.view(output.size(0), -1))
print(output.shape)
output = fc5(output)
print(output.shape)
output = mu(output)
print(output.shape)
