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

| Network Architecture | Year | Original Purpose | Implementation Status |
| ------------------- | ---- | ---------------- | -------------------- |
| LeNet-5 | 1998 | Handwritten digit recognition | ✅ Complete |
| AlexNet | 2012 | ImageNet classification | ✅ Complete |
| Maxout Networks | 2013 | General classification | ✅ Complete |
| NIN (Network in Network) | 2013 | General classification | ✅ Complete |
| VGG (A/B/C/D/E) | 2014 | ImageNet classification | ✅ Complete |
| GoogLeNet (Inception-v1) | 2014 | ImageNet classification | ❌ Planned |
| Inception-v2 | 2015 | ImageNet classification | ❌ Planned |
| Inception-v3 | 2015 | ImageNet classification | ❌ Planned |
| Inception-v4 | 2016 | ImageNet classification | ❌ Planned |
| ResNet | 2015 | ImageNet classification | ❌ Planned |
| MobileNet | 2017 | Mobile vision applications | ❌ Planned |
| EfficientNet | 2019 | Efficient scaling | ❌ Planned |

---
## 🧩 Supported Models & Datasets

### 📊 Paper Original Datasets

| Model | Dataset | Status |
| ----- | ------- | ------ |
| LeNet-5 | MNIST | ✅ Done |
| AlexNet | ImageNet (ILSVRC-2010) | 🔄 Untested |
| VGG | ImageNet (ILSVRC-2012) | 🚧 In Progress |
| VGG | PASCAL VOC (2007/2012) | 🚧 In Progress |
| VGG | Caltech (101/256) | 🚧 In Progress |


*Note: ILSVRC (ImageNet Large Scale Visual Recognition Challenge) is an annual competition that uses a subset of the full ImageNet database. Different years (ILSVRC-2010, ILSVRC-2012, etc.) may use slightly different configurations of the dataset.

### 🔬 Additional Test Datasets

| Model   | Dataset | Compatibility | Status |
| ------- | ------- | ------------ | ------ |
| LeNet-5 | FashionMNIST | ⭐⭐⭐ Excellent | ✅ Done |
| AlexNet | MNIST    | ⭐ Overdesigned | ✅ Done |
| AlexNet | FashionMNIST | ⭐⭐ Adequate | ✅ Done |
| AlexNet | CIFAR10 | ⭐⭐⭐ Excellent | ✅ Done |
| AlexNet | TinyImageNet | ⭐⭐⭐ Excellent | ✅ Done |
| VGG-A   | CIFAR10 | ⭐⭐ Adequate | 🔄 Untested |
| VGG-B   | CIFAR10 | ⭐⭐ Adequate | 🔄 Untested |
| VGG-D(VGG16)  | CIFAR10 | ⭐ Overdesigned | 🔄 Untested |

---
## 📚 References

### Classic Network Architectures
- [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (LeCun et al., 1998)
- [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) (Krizhevsky et al., 2012)
- [VGG](https://arxiv.org/pdf/1409.1556.pdf) (Simonyan & Zisserman, 2014)
- [Inception/GoogleNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) (Szegedy et al., 2015)

### Datasets
- [MNIST](http://yann.lecun.com/exdb/mnist/) (LeCun et al., 2010)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) (Krizhevsky & Hinton, 2009)
- [ImageNet](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf) (Deng et al., 2009)

### more
See [full references list](docs/references.md) for more details.

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

