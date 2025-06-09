import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary as torchinfo_summary

# LeNet-5 architecture
# Reference: https://en.wikipedia.org/wiki/LeNet
# paper: https://ieeexplore.ieee.org/document/726791

class LeNet5(nn.Module):
    def __init__(self, input_shape=(1, 32, 32)):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 原文是降采样，这里用max pooling代替
        # 原文这两层中间使用了稀疏连接，现在这种方式已经废弃，这里没有复现
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten5 = nn.Flatten()

        # 动态计算展开后的特征数
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)      # batch_size=1
            x = self.pool2(F.tanh(self.conv1(dummy_input)))
            x = self.pool4(F.tanh(self.conv3(x)))
            x = self.flatten5(x)
            n_features = x.shape[1]  # 展平后特征数
        
        # self.fc6 = nn.Linear(in_features=16 * 4 * 4, out_features=120) # 这样写死仅用于1*32*32的数据集
        self.fc6 = nn.Linear(n_features, 120)
        self.fc7 = nn.Linear(in_features=120, out_features=84)
        self.fc8 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool2(F.tanh(self.conv1(x)))
        x = self.pool4(F.tanh(self.conv3(x)))
        x = self.flatten5(x)
        x = F.tanh(self.fc6(x))
        x = F.tanh(self.fc7(x))
        x = self.fc8(x)     # pytorch 会自动使用softmax作为最后一层的激活函数
        return x
    
    # 使用torchinfo来打印模型的结构和参数信息
    def summary(self, batch_size=1, input_shape=(1, 32, 32)):
        return torchinfo_summary(
            self, 
            input_size=(batch_size,) + input_shape,  # 添加批量大小
            depth=5, 
            col_names=["input_size", "output_size", "num_params", "kernel_size"]
        )
    
if __name__ == "__main__":
    model = LeNet5(input_shape=(1, 32, 32))
    print(model.summary())

