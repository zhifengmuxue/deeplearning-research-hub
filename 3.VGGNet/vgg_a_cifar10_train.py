from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from VGG import VGG_A
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.train_utils import run_training


def get_dataloaders(dataset_name, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   
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

run_training(
    model_class=VGG_A,
    get_dataloaders_fn=get_dataloaders,
    dataset_name="CIFAR10",
    input_shape=(3, 224, 224),
    num_epochs=1,
    batch_size=128,
    output_dir="3.VGGNet/outputs",
    enable_plot=True,
)
