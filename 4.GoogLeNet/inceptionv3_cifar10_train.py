from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from InceptionV3 import InceptionV3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.train_utils import run_training

def get_dataloaders(dataset_name, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),     # 放大图像
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

model = InceptionV3(input_shape=(3, 299, 299), num_classes=10)
run_training(
    model,
    get_dataloaders_fn=get_dataloaders,
    dataset_name="CIFAR10",
    num_epochs=5,
    batch_size=64,
    output_dir="4.GoogLeNet/outputs",
    enable_plot=True,
)
