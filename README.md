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

| Model   | Dataset | Compatibility | Original Paper Dataset | Status |
| ------- | ------- | ------------ | --------------------- | ------ |
| LeNet-5 | MNIST   | ⭐⭐⭐ Excellent | ✅ Yes | ✅ Done |
| LeNet-5 | FashionMNIST | ⭐⭐⭐ Excellent | ❌ No | ✅ Done |
| AlexNet | MNIST    | ⭐ Overdesigned | ❌ No | ✅ Done |
| AlexNet | FashionMNIST | ⭐⭐ Adequate | ❌ No | ✅ Done |
| AlexNet | CIFAR10 |  ⭐⭐⭐ Excellent | ❌ No | ✅ Done |
| AlexNet | ILSVRC-2010 | ⭐⭐⭐ Excellent | ✅ Yes | 🚧 Planned |
| AlexNet | ImageNet | ⭐⭐⭐ Excellent | ✅ Yes | 🚧 Planned |
| AlexNet | TinyImageNet | ⭐⭐⭐ Excellent | ❌ No | ✅ Done |

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

