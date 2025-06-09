import torch
import torch.nn as nn
from torchinfo import summary as torchinfo_summary

# LeNet-5 architecture
# Reference: https://en.wikipedia.org/wiki/LeNet
# paper: https://ieeexplore.ieee.org/document/726791

class LeNet5(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), num_classes=10):
        super(LeNet5, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),  # 28x28
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # 10x10
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 5x5
        )

        # 动态计算展平后特征数量
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.features(dummy_input)
            n_features = x.view(1, -1).shape[1]

        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    # 打印模型结构
    def summary(self, batch_size=1, input_shape=(1, 32, 32)):
        return torchinfo_summary(
            self,
            input_size=(batch_size,) + input_shape,
            depth=5,
            col_names=["input_size", "output_size", "num_params", "kernel_size"]
        )
    


    
if __name__ == "__main__":
    model = LeNet5(input_shape=(1, 32, 32))
    print(model.summary())

