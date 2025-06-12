import torch
import torch.nn as nn
from torchinfo import summary as torchinfo_summary

# InceptionV3 网络
# Inception v3 已经更常作为一个完整独立的 end-to-end 网络架构来用,
# 而不再是像 v1 时那样的简单 block 级模块

# 这个块 也就是InceptionV2
class InceptionA(nn.Module):
    def __init__(self, in_channels, 
                 out_1x1,
                 out_5x5_reduce, out_5x5_1, out_5x5_2,
                 out_3x3_reduce, out_3x3_1, out_3x3_2,
                 pool_proj):
        super(InceptionA, self).__init__()

        # 1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> 3x3 -> 3x3 (替代原 5x5)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(out_5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5_reduce, out_5x5_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_5x5_1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5_1, out_5x5_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_5x5_2),
            nn.ReLU(inplace=True),
        )

        # 1x1 -> 3x3 -> 3x3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(out_3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_3x3_reduce, out_3x3_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3_1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_3x3_1, out_3x3_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3_2),
            nn.ReLU(inplace=True),
        )

        # AvgPool -> 1x1
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1x1 = self.branch1(x)
        branch5x5 = self.branch2(x)
        branch3x3 = self.branch3(x)
        branch_pool = self.branch4(x)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)
    
    def summary(self, batch_size=1, input_shape=(3, 299, 299)):
        return torchinfo_summary(
            self,
            input_size=(batch_size, *input_shape),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=3,
            device=torch.device("cpu")
        )


class InceptionB(nn.Module):
    def __init__(self, in_channels, out_channels=192, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2  # 保持尺寸不变的padding

        # 2次因式分解卷积
        # 1x1 -> 1xn -> nx1 -> 1xn -> nx1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 1次因式分解卷积
        # 1x1 -> 1xn -> nx1
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 平均池化分支
        # pool -> 1x1 conv
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 1x1 卷积分支
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
    
    def summary(self, batch_size=1, input_shape=(3, 299, 299)):
        return torchinfo_summary(
            self,
            input_size=(batch_size, *input_shape),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=3,
            device=torch.device("cpu")
        )


class InceptionC(nn.Module):
    def __init__(self, in_channels,
                 out_branch1=320,
                 out_branch2=384,
                 out_branch2a=384,
                 out_branch2b=384,
                 out_branch3_reduce=448,
                 out_branch3=384,
                 out_branch3a=384,
                 out_branch3b=384,
                 out_branch4=192):
        super().__init__()

        # 1x1 Conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_branch1, kernel_size=1),
            nn.BatchNorm2d(out_branch1),
            nn.ReLU(inplace=True),
        )

        # 1x1 -> 1x3(2a)
        # 1x1 -> 3x1(2b)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_branch2, kernel_size=1),
            nn.BatchNorm2d(out_branch2),
            nn.ReLU(inplace=True),
        )
        self.branch2a = nn.Conv2d(out_branch2, out_branch2a, kernel_size=(1, 3), padding=(0, 1))
        self.branch2b = nn.Conv2d(out_branch2, out_branch2b, kernel_size=(3, 1), padding=(1, 0))

        # 1x1 -> 3x3 -> 1x3(3a)
        # 1x1 -> 3x3 -> 3x1(3b)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_branch3_reduce, kernel_size=1),
            nn.BatchNorm2d(out_branch3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_branch3_reduce, out_branch3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_branch3),
            nn.ReLU(inplace=True),
        )
        self.branch3a = nn.Conv2d(out_branch3, out_branch3a, kernel_size=(1, 3), padding=(0, 1))
        self.branch3b = nn.Conv2d(out_branch3, out_branch3b, kernel_size=(3, 1), padding=(1, 0))

        # pool -> 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_branch4, kernel_size=1),
            nn.BatchNorm2d(out_branch4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2(x)
        branch2 = torch.cat([self.branch2a(branch2), self.branch2b(branch2)], 1)

        branch3 = self.branch3(x)
        branch3 = torch.cat([self.branch3a(branch3), self.branch3b(branch3)], 1)

        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
    
    def summary(self, batch_size=1, input_shape=(3, 299, 299)):
        return torchinfo_summary(
            self,
            input_size=(batch_size, *input_shape),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=3,
            device=torch.device("cpu")
        )
    
class InceptionV3(nn.Module):
    """
        | 操作层级                   | 输出尺寸       | 说明与备注                          |
        | ---------------------- | ---------- | ------------------------------ |
        | 输入图像                   | 299x299x3  | RGB图像                          |
        | Conv 3x3 / stride 2    | 149x149x32 | 无填充,步幅2,减小尺寸                   |
        | Conv 3x3 / stride 1    | 147x147x32 | 有填充,保持尺寸                       |
        | Conv padded 3x3 / 1    | 147x147x64 | 有填充                            |
        | MaxPool 3x3 / stride 2 | 73x73x64   | 空间尺寸减半                         |
        | Conv 3x3 / stride 1    | 71x71x80   | 无填充,尺寸稍微减小                     |
        | Conv 3x3 / stride 2    | 35x35x192  | 无填充,尺寸减半                       |
        | MaxPool 3x3 / stride 2 | 35x35x192  | 论文中尺寸保持35x35,但进行了池化操作          |
        | 3 * Inception A 模块  | 35x35x288  | 3个Inception A模块堆叠              |
        | 5 * Inception B 模块  | 17x17x768  | 5个Inception B模块堆叠,包含尺寸缩减       |
        | 2 * Inception C 模块  | 8x8x1280   | 2个Inception C模块堆叠,进一步抽象特征      |
        | AvgPool (8x8)          | 1x1x2048   | 全局平均池化                         |
        | Dropout + FC           | 1x1x1000   | 全连接层输出1000类logits（ImageNet分类数） |
        | Softmax                | 1x1x1000   | 分类概率输出                         |
    """
    def __init__(self, num_classes=1000,
                 input_shape=(3, 299, 299),
                 stem_out_channels=(32, 32, 64, 80, 192),
                 inception_a_channels=(64, 48, 64, 64, 64, 96, 96, 32),
                 inception_b_out_channels=192,
                 inception_c_channels=(320, 384, 384, 384, 448, 384, 384, 384, 192),
                 dropout_prob=0.5):
        super().__init__()
        input_channels = input_shape[0]
        c1, c2, c3, c4, c5 = stem_out_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=3, stride=2),  # 299->149
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),  # 149->147
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),  # 147->147
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2),  # 147->73

            nn.Conv2d(c3, c4, kernel_size=1),  # 73->71
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),

            nn.Conv2d(c4, c5, kernel_size=3),  # 71->35
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2),  # 35->17
        )

        # 计算 InceptionA 输出通道数
        out_1x1 = inception_a_channels[0]
        out_5x5_2 = inception_a_channels[3]
        out_3x3_2 = inception_a_channels[6]
        pool_proj = inception_a_channels[7]
        inception_a_out_channels = out_1x1 + out_5x5_2 + out_3x3_2 + pool_proj  # 64+64+96+32=256

        # 3 个 Inception A 模块
        self.inception_a1 = InceptionA(c5, *inception_a_channels)  # 输入c5=192
        self.inception_a2 = InceptionA(inception_a_out_channels, *inception_a_channels)  # 输入256
        self.inception_a3 = InceptionA(inception_a_out_channels, *inception_a_channels)  # 输入256

        self.reduction_a = nn.MaxPool2d(3, stride=2)  # 17->8 (池化不改变通道数)

        # InceptionB 输出通道数
        inception_b_out_channels = inception_b_out_channels * 4  # 192*4=768

        # 5 个 Inception B 模块
        self.inception_b1 = InceptionB(inception_a_out_channels)  # 输入256，输出768
        self.inception_b2 = InceptionB(inception_b_out_channels)  # 输入768
        self.inception_b3 = InceptionB(inception_b_out_channels)
        self.inception_b4 = InceptionB(inception_b_out_channels)
        self.inception_b5 = InceptionB(inception_b_out_channels)

        self.reduction_b = nn.MaxPool2d(3, stride=2)  # 8->4 (池化不改变通道数)

        # 计算 InceptionC 输出通道数
        inception_c_out_channels = (
            inception_c_channels[0] +       # branch1 输出通道
            inception_c_channels[2] * 2 +   # branch2 两个分支输出通道
            inception_c_channels[5] * 2 +   # branch3 两个分支输出通道
            inception_c_channels[8]         # branch4 输出通道
        )

        # 2 个 Inception C 模块
        self.inception_c1 = InceptionC(inception_b_out_channels, *inception_c_channels)  # 输入768
        self.inception_c2 = InceptionC(inception_c_out_channels, *inception_c_channels)  # 输入2176

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(inception_c_out_channels, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.inception_a1(x)
        x = self.inception_a2(x)
        x = self.inception_a3(x)

        x = self.reduction_a(x)

        x = self.inception_b1(x)
        x = self.inception_b2(x)
        x = self.inception_b3(x)
        x = self.inception_b4(x)
        x = self.inception_b5(x)

        x = self.reduction_b(x)

        x = self.inception_c1(x)
        x = self.inception_c2(x)

        x = self.classifier(x)

        return x

    
    def summary(self, batch_size=1, input_shape=(3, 299, 299)):
        return torchinfo_summary(
            self,
            input_size=(batch_size, *input_shape),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=3,
            device=torch.device("cpu")
        )

if __name__ == "__main__":
    # model = InceptionA(
    #     in_channels=3,  # 改成3,和输入数据通道数匹配
    #     out_1x1=64,
    #     out_5x5_reduce=48, out_5x5_1=64, out_5x5_2=64,
    #     out_3x3_reduce=64, out_3x3_1=96, out_3x3_2=96,
    #     pool_proj=32
    # )
    # model.summary(batch_size=1, input_shape=(3, 299, 299))

    # model = InceptionB(in_channels=3, out_channels=192, kernel_size=7)
    # model.summary(batch_size=1, input_shape=(3, 299, 299))

    # model = InceptionC(
    #     in_channels=3,
    #     out_branch1=320,
    #     out_branch2=384,
    #     out_branch2a=384,
    #     out_branch2b=384,
    #     out_branch3_reduce=448,
    #     out_branch3=384,
    #     out_branch3a=384,
    #     out_branch3b=384,
    #     out_branch4=192
    # )
    # model.summary(batch_size=1, input_shape=(3, 299, 299))

    model = InceptionV3(num_classes=1000)
    model.summary(batch_size=1, input_shape=(3, 299, 299))

    pass