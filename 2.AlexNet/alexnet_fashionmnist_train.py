from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from AlexNet import AlexNet
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.train_utils import run_training

# AlexNet 不适合 fashionMNIST 数据集，因为它的输入尺寸较大且通道数为 3。
# 这里用 transforms 将 fashionMNIST 图像转换为 3 通道，并调整大小到 224x224。
# 这种转换虽然不是最佳实践，但可以让 AlexNet 在 fashionMNIST 上运行。
# 实际应用中，建议使用更适合 fashionMNIST 的模型，如 LeNet-5。
def get_dataloaders(dataset_name, batch_size=64):
    # AlexNet 需要较大的输入图像尺寸
    transform = transforms.Compose([
        transforms.Resize((224, 224)),     # AlexNet 要求大图像
        transforms.Grayscale(num_output_channels=3),  # 1 通道变 3 通道
        transforms.ToTensor()
    ])

    if dataset_name.lower() == "fashionmnist":
        train_set = datasets.FashionMNIST(root="./dataset", train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root="./dataset", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

run_training(
    model_class=AlexNet,
    get_dataloaders_fn=get_dataloaders,
    dataset_name="FashionMNIST",
    input_shape=(3, 224, 224),
    num_epochs=1,    # 训练十分慢，这里就调了训练一次
    output_dir="AlexNet/outputs",
    enable_plot=True,
)
