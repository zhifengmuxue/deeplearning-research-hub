# 🧠 DeepLearning Research Hub

A modular and extensible deep learning research framework designed for quick experimentation with classic and modern neural network architectures across various datasets.

## 📁 Project Structure

```
deeplearning-research-hub/
├── LeNet/                     # LeNet5 architecture and training script for MNIST
│   ├── lenet5.py
│   ├── lenet5_minist_train.py
|   ├── lenet5_minist_predict.py
|   └── ... other dataset
|
├── ... other network
|
├── tools/                    # Generic training & evaluation utilities
|   ├── predict_utils.py
│   └── train_utils.py
├── requirements.txt          # Python dependencies
├── README.md                 # You're reading this!
└── train.py (optional)       # Central training entry point
```

---

## 🚀 Features

* ✅ **Model-agnostic training** via `run_training()`
* ✅ **Plug-and-play model & dataset support**
* ✅ **Metrics visualization** (Loss & Accuracy per epoch)
* ✅ **Automatic model saving**
* ✅ **Clean modular code** for easy extension

---

## 🏗️ Implemented Network Architectures

| Network Architecture | Year | Original Purpose | Implementation Status | Notes |
| ------------------- | ---- | ---------------- | -------------------- | ----- |
| LeNet-5 | 1998 | Handwritten digit recognition | ✅ Complete | 经典CNN架构，为现代卷积网络奠定基础 |
| AlexNet | 2012 | ImageNet classification | ✅ Complete | 首个深度CNN赢得ImageNet竞赛，引发深度学习革命 |
| VGG (A/B/C/D/E) | 2014 | ImageNet classification | 🚧 In Progress | 以简洁的架构和3x3卷积堆叠著称 |
| GoogleNet/Inception | 2014 | ImageNet classification | ❌ Planned | 引入inception模块，减少参数量 |
| ResNet | 2015 | ImageNet classification | ❌ Planned | 解决深层网络的梯度消失问题，引入残差连接 |
| MobileNet | 2017 | Mobile vision applications | ❌ Planned | 轻量级网络，适用于移动设备 |
| EfficientNet | 2019 | Efficient scaling | ❌ Planned | 通过复合缩放方法平衡网络宽度、深度和分辨率 |

---
## 🧩 Supported Models & Datasets

### 📊 Paper Original Datasets

| Model   | Dataset | Compatibility | Status | Notes |
| ------- | ------- | ------------ | ------ | ----- |
| LeNet-5 | MNIST   | ⭐⭐⭐ Excellent | ✅ Done | 论文原始使用的手写数字数据集 |
| AlexNet | ILSVRC-2010 | ⭐⭐⭐ Excellent | 🚧 Planned | 原始ImageNet竞赛数据集 |
| AlexNet | ImageNet | ⭐⭐⭐ Excellent | 🚧 Planned | 完整ImageNet数据集 |
| VGG-16  | ILSVRC-2012 | ⭐⭐⭐ Excellent | 🚧 In Progress | 论文中表现最佳的VGG变体 |
| VGG-19  | ILSVRC-2012 | ⭐⭐⭐ Excellent | 🚧 In Progress | 论文中最深的VGG变体 |


### 🔬 Additional Test Datasets

| Model   | Dataset | Compatibility | Status | Purpose |
| ------- | ------- | ------------ | ------ | ------- |
| LeNet-5 | FashionMNIST | ⭐⭐⭐ Excellent | ✅ Done | 测试模型在类似结构但不同内容数据上的表现 |
| AlexNet | MNIST    | ⭐ Overdesigned | ✅ Done | 验证复杂模型在简单任务上的过拟合情况 |
| AlexNet | FashionMNIST | ⭐⭐ Adequate | ✅ Done | 测试深度模型在简单灰度图像上的适应性 |
| AlexNet | CIFAR10 | ⭐⭐⭐ Excellent | ✅ Done | 评估在小型彩色图像数据集上的表现 |
| AlexNet | TinyImageNet | ⭐⭐⭐ Excellent | ✅ Done | ImageNet的简化版本，减少类别和图像数量 |
| VGG-A   | CIFAR10 | ⭐⭐ Adequate | 🚧 In Progress | 测试最简单的VGG变体在小型数据集上的性能 |
| VGG-B   | CIFAR10 | ⭐⭐ Adequate | 🚧 In Progress | 评估额外卷积层对性能的影响 |
| VGG-D   | CIFAR10 | ⭐ Overdesigned | 🚧 In Progress | 测试深层VGG在小型数据集上的过拟合情况 |

> ✅ More coming soon: GoogleNet, ResNet, CIFAR10...


---

## 🛠️ Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/zhifengmuxue/deeplearning-research-hub.git
cd deeplearning-research-hub
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # on Windows use `venv\Scripts\activate`
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Run a Training Script (Example: LeNet5 on MNIST)

```bash
cd LeNet
python lenet5_minist_train.py
```

---

## 📦 Customize Your Own Training

You can create new scripts like this:

```python
# custom_train.py
from tools.train_utils import run_training
from LeNet import LeNet5
from tools.datasets import get_mnist_dataloaders  # or your own dataset function

run_training(
    model_class=LeNet5,
    get_dataloaders_fn=get_mnist_dataloaders,
    dataset_name="MNIST",
    input_shape=(1, 32, 32),
    lr=0.001,
    batch_size=64,
    num_epochs=10,
    output_dir="outputs"
)
```

---

## 📊 Visual Output

After training, metrics are automatically plotted and saved under:

```
outputs/
├── visuals/    # PNG line plots of Loss & Accuracy
└── weights/    # Saved model weights
```


---

## 🤝 Contributing

Pull requests are welcome! Whether it’s adding a new dataset, improving training utilities, or implementing a new model, feel free to fork and contribute.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙋 FAQ

**Q: Why is `tools` not found?**
A: Run scripts from the project root or ensure `tools/` is on `PYTHONPATH`. Alternatively, use relative imports.

**Q: How to extend to a new dataset?**
A: Just define a `get_dataloaders_fn(dataset_name, batch_size)` function, return train/test loaders.

