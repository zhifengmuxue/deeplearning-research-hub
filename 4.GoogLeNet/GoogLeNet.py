import torch
import torch.nn as nn
from InceptionV1 import InceptionV1
from InceptionV2 import InceptionV2
from torchinfo import summary as torchinfo_summary

# GoogLeNet(Inceptionv1) 模型
# 采用 Auxiliary Classifier 辅助：防止信息过度丢失
# mps暂不支持 LRN 的 avg_pool3d，用BN 替换LRN
# 同时AdaptiveAvgPool2d 来替换原本的全局平均池化，以适应不同尺寸的图片

class GoogLeNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000, version="v1"):
        super(GoogLeNet, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        version = version.lower()

        match version:
            case "v1":
                InceptionBlock = InceptionV1
            case "v2":
                InceptionBlock = InceptionV2
            case _:
                raise ValueError(f"Unsupported version: {version}")


        self.stage1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.LocalResponseNorm(64, alpha=0.0001, beta=0.75, k=2),
            nn.BatchNorm2d(64), # 替换 LRN
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.LocalResponseNorm(192, alpha=0.0001, beta=0.75, k=2),
            nn.BatchNorm2d(192), # 替换 LRN
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(480, 192, 96, 208, 16, 48, 64),
        )
        self.aux1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # 自动池化成 4x4
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

        self.stage2 = nn.Sequential(
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),
        )
        self.aux2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # 自动池化成 4x4
            nn.Conv2d(528, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

        self.stage3 = nn.Sequential(
            InceptionBlock(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(832, 256, 160, 320, 32, 128, 128),
            InceptionBlock(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(1024, num_classes),  # 最终分类器
        )

    def forward(self, x):
        x = self.stage1(x)
        aux1 = self.aux1(x)
        x = self.stage2(x)
        aux2 = self.aux2(x)
        x = self.stage3(x)

        if self.training:
            return x, aux1, aux2
        else:
            return x   # 训练时返回主分类器和两个辅助分类器的输出 测试时只需要主分类器
    
    # 损失函数 0.3是论文中的取值
    def compute_loss(self, outputs, labels, criterion):
        main_output, aux1_output, aux2_output = outputs
        loss_main = criterion(main_output, labels)
        loss_aux1 = criterion(aux1_output, labels)
        loss_aux2 = criterion(aux2_output, labels)
        return loss_main + 0.3 * (loss_aux1 + loss_aux2)

        
    def summary(self, batch_size=1):
        return torchinfo_summary(
            self,
            input_size=(batch_size, *self.input_shape),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=3,
            device=torch.device("cpu")
        )
    
if __name__ == "__main__":
    model = GoogLeNet(input_shape=(3, 224, 224), num_classes=1000,version="v2")
    model.summary(batch_size=1)
    # 测试模型输出
    x = torch.randn(1, 3, 224, 224)
    outputs = model(x)
    print("主分类器输出:", outputs[0].shape)
    print("辅助分类器1输出:", outputs[1].shape)
    print("辅助分类器2输出:", outputs[2].shape)

