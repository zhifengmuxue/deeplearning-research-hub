from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from AlexNet import AlexNet
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.train_utils import run_training

# AlexNet 检测 ImageNet 数据集 (原文)
# 注意：ImageNet 数据集非常大，下载和训练可能需要较长时间。
# 确保你有足够的存储空间和计算资源。
# 如果只想体验一下，可以采用tiny_imagenet 数据集，
# 它是 ImageNet 的一个小型版本，包含 200 个类别，每个类别有 500 张训练图像和 50 张验证图像。
"""
Tiny ImageNet 数据集下载地址：
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -O ./dataset/tiny-imagenet-200.zip
解压数据集
unzip ./dataset/tiny-imagenet-200.zip -d ./dataset/
"""

def get_dataloaders(dataset_name, batch_size=64):
    # AlexNet 需要较大的输入图像尺寸
    transform = transforms.Compose([
        transforms.Resize((224, 224)),     # AlexNet 要求大图像
        transforms.ToTensor()
    ])

    if dataset_name.lower() == "tinyimagenet":
        train_set = datasets.ImageFolder(root="./dataset/tiny-imagenet-200/train", transform=transform)
        test_set = datasets.ImageFolder(root="./dataset/tiny-imagenet-200/val", transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

model = AlexNet(input_shape=(3, 224, 224), num_classes=200)  # TinyImageNet 有 200 个类别
run_training(
    model,
    get_dataloaders_fn=get_dataloaders,
    dataset_name="TinyImageNet", 
    num_epochs=1,    # 训练十分慢，这里就调了训练一次
    batch_size=64,
    output_dir="2.AlexNet/outputs",
    enable_plot=True,
)
