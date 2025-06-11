import torch
import torch.nn as nn
from torchinfo import summary as torchinfo_summary
# 各个版本的 Inception 模块

class InceptionV1(nn.Module):
    def __init__(self,in_channels, out_channels1x1, out_channels3x3red, 
                 out_channels3x3, out_channels5x5red, out_channels5x5, pool_proj):
        super(InceptionV1, self).__init__()

        # 1x1 conv
        self.branch1 = nn.Conv2d(in_channels, out_channels1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels3x3red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels3x3red, out_channels3x3, kernel_size=3, padding=1)
        )

        # 1x1 conv -> 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels5x5red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels5x5red, out_channels5x5, kernel_size=5, padding=2)
        )

        # 3x3 pool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
    
    def summary(self, batch_size=1, input_shape=(3, 224, 224)):

        return torchinfo_summary(
            self,
            input_size=(batch_size, *input_shape),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=3,
            device=torch.device("cpu")
        )


class GoogLeNetV1Tiny(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000):
        super(GoogLeNetV1Tiny, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # 定义 GoogLeNet 的 Inception 模块
        self.inception1 = InceptionV1(3, 64, 96, 128, 16, 32, 32)
        self.inception2 = InceptionV1(256, 128, 128, 192, 32, 96, 64)
        
        # 全局平均池化和分类器
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(480, num_classes)  # 输出通道数为480

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def summary(self, batch_size=1):
        return torchinfo_summary(
            self,
            input_size=(batch_size, *self.input_shape),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=3,
            device=torch.device("cpu")
        )
    

if __name__ == "__main__":


    # 创建一个 InceptionV1 模型实例
    model = InceptionV1(
        in_channels=3, 
        out_channels1x1=64, 
        out_channels3x3red=96, 
        out_channels3x3=128, 
        out_channels5x5red=16, 
        out_channels5x5=32, 
        pool_proj=32
    )
    
    # 打印模型摘要
    model.summary()

    model = GoogLeNetV1Tiny(input_shape=(3, 224, 224), num_classes=1000)
    print(model.summary(batch_size=1))