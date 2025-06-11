import torch
import torch.nn as nn
from torchinfo import summary as torchinfo_summary

# Network In Network (NIN) 模型
# 用于拟合任意函数

class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class NIN(nn.Module):
    def __init__(self, input_shape=[1, 28, 28], num_classes=10):
        super().__init__()
        C, H, W = input_shape

        self.features = nn.Sequential(
            # 第一组 MLPConv
            nn.Conv2d(C, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            # 第二组 MLPConv
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14 -> 7x7

            # 第三组 MLPConv
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 全连接层可以不需要，NIN通常用GAP直接输出类别数
        # 如果想用分类器，可以做简单flatten+linear
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_classes, num_classes)  # 这里相当于一个线性分类器
        )

    def forward(self, x):
        x = self.features(x)        
        x = self.classifier(x)       
        return x
    
    def summary(self, batch_size=1, input_shape=(1, 28, 28)):
        return torchinfo_summary(
            self,
            input_size=(batch_size,) + input_shape,
            depth=5,
            col_names=["input_size", "output_size", "num_params", "kernel_size"]
        )
    
if __name__ == "__main__":
    model = NIN(input_shape=(1, 28, 28), num_classes=10)
    print(model.summary(batch_size=1, input_shape=(1, 28, 28)))
