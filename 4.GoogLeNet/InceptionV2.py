import torch
import torch.nn as nn
from torchinfo import summary as torchinfo_summary
# inception v2 模块

class InceptionV2(nn.Module):
    def __init__(self, in_channels, out_channels1x1, out_channels3x3red, 
                 out_channels3x3, out_channels5x5red, out_channels5x5, pool_proj):
        super(InceptionV2, self).__init__()

        # 1x1 卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1x1, kernel_size=1),
            nn.BatchNorm2d(out_channels1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 卷积 -> 3x3 卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels3x3red, kernel_size=1),
            nn.BatchNorm2d(out_channels3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels3x3red, out_channels3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1 卷积 -> 3x3 卷积 -> 3x3 卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels5x5red, kernel_size=1),
            nn.BatchNorm2d(out_channels5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels5x5red, out_channels5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels5x5, out_channels5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3 池化 -> 1x1 卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
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

if __name__ == "__main__":
    # 测试 InceptionV2 模块
    model = InceptionV2(3, 64, 96, 128, 16, 32, 32)
    model.summary(batch_size=1, input_shape=(3, 224, 224))
    