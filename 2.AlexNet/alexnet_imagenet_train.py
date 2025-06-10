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

def get_dataloaders(dataset_name, batch_size=64):
    # AlexNet 需要较大的输入图像尺寸
    transform = transforms.Compose([
        transforms.Resize((224, 224)),     # AlexNet 要求大图像
        transforms.ToTensor()
    ])

    if dataset_name.lower() == "imagenet":
        train_set = datasets.ImageNet(root="./dataset", split='train', download=True, transform=transform)
        test_set = datasets.ImageNet(root="./dataset", split='val', download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

run_training(
    model_class=AlexNet,
    get_dataloaders_fn=get_dataloaders,
    dataset_name="ImageNet", 
    input_shape=(3, 224, 224),
    num_epochs=1,    # 训练十分慢，这里就调了训练一次
    batch_size=64,
    output_dir="2.AlexNet/outputs",
    enable_plot=True,
)
