import torch
import torch.nn as nn
from torchinfo import summary as torchinfo_summary

# AlexNet architecture
# Reference: https://en.wikipedia.org/wiki/VGGNet
# paper: VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
# 本脚本中包含论文中出现的各种结构 （vgg-a ~ vgg-e）

# 论文中的vgg-a (11层)
class VGG_A(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000):
        super(VGG_A, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),  # 输出: (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (64, 112, 112)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 输出: (128, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (128, 56, 56)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (256, 28, 28)

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (512, 14, 14)

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 输出: (512, 7, 7)
        )

        # 动态计算展平后特征数量
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape) 
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
    
# 论文中的vgg-a (11层) + LRN , 论文结果证明LRN仅仅消耗了算力，而没有带来性能提升。    
class VGG_A_LRN(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000):
        super(VGG_A_LRN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),  # 输出: (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),  # LRN
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (64, 112, 112)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 输出: (128, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (128, 56, 56)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (256, 28, 28)

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (512, 14, 14)

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 输出: (512, 7, 7)
        )

        # 动态计算展平后特征数量
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape) 
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
        
# 论文中的vgg-b (13层)       
class VGG_B(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000):
        super(VGG_B, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),  # 输出: (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # 输出: (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (64, 112, 112)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 输出: (128, 112, 112)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),  # 输出: (128, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (128, 56, 56)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (256, 28, 28)

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (512, 14, 14)

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 输出: (512, 7, 7)
        )

        # 动态计算展平后特征数量
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape) 
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
    
# 论文中的vgg-c (16层)
class VGG_C(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000):
        super(VGG_C, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),  # 输出: (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # 输出: (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (64, 112, 112)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 输出: (128, 112, 112)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),  # 输出: (128, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (128, 56, 56)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (256, 28, 28)

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (512, 14, 14)

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 输出: (512, 7, 7)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape) 
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
    
# 论文中的vgg-d (16层) 也被称为vgg16
class VGG16(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),  # 输出: (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # 输出: (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (64, 112, 112)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 输出: (128, 112, 112)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),  # 输出: (128, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (128, 56, 56)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (256, 28, 28)

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28) 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (512, 14, 14)

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 输出: (512, 7, 7)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape) 
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

# 论文中的vgg-e (19层)
class VGG19(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),  # 输出: (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # 输出: (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (64, 112, 112)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 输出: (128, 112, 112)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),  # 输出: (128, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (128, 56, 56)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),  # 输出: (256, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (256, 28, 28)

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (512, 14, 14)

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 输出: (512, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 输出: (512, 7, 7)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape) 
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