import torch
import torch.nn as nn
from torchinfo import summary as torchinfo_summary

# AlexNet architecture
# Reference: https://en.wikipedia.org/wiki/AlexNet
# paper: 《ImageNet Classification with Deep Convolutional Neural Networks》

class AlexNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000):
        super(AlexNet, self).__init__()
        # 原文的归一化采用的是local response normalization (LRN)，
        # 但在vgg的论文中被指出没有什么效果
        # 我们更通常使用 Batch Normalization 替代。
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),  # 输出: (96, 55, 55)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),       # 输出: (96, 27, 27)

            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # 输出: (256, 27, 27)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),         # 输出: (256, 13, 13)

            nn.Conv2d(256, 384, kernel_size=3, padding=1), # 输出: (384, 13, 13)
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1), # 输出: (384, 13, 13)
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 输出: (256, 13, 13)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)          # 输出: (256, 6, 6)
        )

        # 动态计算展平后特征数量
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # 例如 (1, 32, 32)
            x = self.features(dummy_input)              # 所有 Conv 和 Pooling 层
            x = torch.flatten(x, 1)
            n_features = x.shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def summary(self, batch_size=1, input_shape=(3, 224, 224)):
        return torchinfo_summary(
            self,
            input_size=(batch_size,) + input_shape,
            depth=5,
            col_names=["input_size", "output_size", "num_params", "kernel_size"]
        )
    
if __name__ == "__main__":
    model = AlexNet(input_shape=(3, 224, 224))
    print(model.summary())
