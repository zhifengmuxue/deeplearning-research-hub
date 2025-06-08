import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import datetime

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, log_interval=100):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (batch_idx + 1) % log_interval == 0:
            print(f"Epoch [{epoch}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {running_loss / log_interval:.6f}")
            running_loss = 0.0
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def plot_metrics(losses, accuracies, output_dir):
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'b-o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, 'r-o')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"training_metrics_{timestamp}.png")
    plt.savefig(save_path)
    print(f"Training metrics plot saved to: {save_path}")
    plt.show()

def run_training(
    model_class,
    get_dataloaders_fn,
    dataset_name="MNIST",
    input_shape=(1, 32, 32),
    lr=0.001,
    batch_size=64,
    num_epochs=5,
    output_dir="outputs",
    enable_plot=True,
    device=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model_class(input_shape=input_shape).to(device)
    train_loader, test_loader = get_dataloaders_fn(dataset_name, batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    accuracies = []

    for epoch in range(1, num_epochs + 1):
        print(f"Start Epoch {epoch}")
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        acc = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {acc:.2f}%")
        losses.append(avg_loss)
        accuracies.append(acc)
        print(f"End Epoch {epoch}\n")

    if enable_plot:
        plot_metrics(losses, accuracies, output_dir=os.path.join(output_dir, "visuals"))

    os.makedirs(os.path.join(output_dir, "weights"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "weights", f"{model_class.__name__}_{dataset_name}.pth"))
    print("Model saved.")

