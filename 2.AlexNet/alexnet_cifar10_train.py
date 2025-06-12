from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from AlexNet import AlexNet
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.train_utils import run_training

# 运行 AlexNet 在 CIFAR-10 上的训练脚本
# 注意：CIFAR-10 图像较小(32×32)，AlexNet 可能不是最佳选择，但可以用于实验。
# 运行后会在 "AlexNet/outputs" 目录下生成训练结果和权重文件。
# 你可以根据需要调整 batch_size 和 num_epochs。
def get_dataloaders(dataset_name, batch_size=64):
    # AlexNet 需要较大的输入图像尺寸
    transform = transforms.Compose([
        transforms.Resize((224, 224)),     # AlexNet 要求大图像
        transforms.Grayscale(num_output_channels=3),  # 1 通道变 3 通道
        transforms.ToTensor()
    ])

    if dataset_name.lower() == "cifar10":
        train_set = datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

model = AlexNet(input_shape=(3, 224, 224), num_classes=10)
run_training(
    model,
    get_dataloaders_fn=get_dataloaders,
    dataset_name="CIFAR10",
    num_epochs=5,
    batch_size=64,
    output_dir="2.AlexNet/outputs",
    enable_plot=True,
)
