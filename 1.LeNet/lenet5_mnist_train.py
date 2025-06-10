from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from LeNet import LeNet5
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.train_utils import run_training

def get_dataloaders(dataset_name, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    if dataset_name.lower() == "mnist":
        train_set = datasets.MNIST(root="./dataset", train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root="./dataset", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


run_training(
    model_class=LeNet5,
    get_dataloaders_fn=get_dataloaders,
    dataset_name="MNIST",
    input_shape=(1, 28, 28),
    num_epochs=5,
    output_dir="1.LeNet/outputs",
    enable_plot=True,
)
