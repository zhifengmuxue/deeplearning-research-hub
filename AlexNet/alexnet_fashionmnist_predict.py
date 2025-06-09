from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from AlexNet import AlexNet
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.predict_utils import predict_single_image

### Alex fashionMNIST 单图像预测脚本
### 先训练权重，在进行预测
### 这个是衣物分类（灰度图）

label_map = {
    0: "上衣",
    1: "裤子",
    2: "套头衫",
    3: "连衣裙",
    4: "外套",
    5: "凉鞋",
    6: "衬衫",
    7: "运动鞋",
    8: "包",
    9: "踝靴"
}

# 定义transform（要与训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),     # AlexNet 要求大图像
    transforms.Grayscale(num_output_channels=3),  # 1 通道变 3 通道
    transforms.ToTensor()
])

# 加载fashionMNIST测试集
test_dataset = datasets.FashionMNIST(root='./dataset', train=False, download=True, transform=transform)

random.seed(99)

# 随机获取图像和标签
random_idx = random.randint(0, len(test_dataset) - 1)
img_tensor, label = test_dataset[random_idx]

# 打印选择的索引，便于后续验证
print(f"随机选择的样本索引: {random_idx}")

img_to_show = img_tensor.permute(1, 2, 0).numpy()  # 调整维度顺序

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.imshow(img_to_show)  
plt.title(f"True Label: {label_map[label]}")  # 使用 label_map 显示类别名称
plt.axis("off")
plt.show()

# 初始化模型
model = AlexNet(input_shape=(3, 224, 224))

# 权重路径
current_dir = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(current_dir, "outputs", "weights", "AlexNet_FashionMNIST.pth")

# 预测
predicted_class, predicted_prob = predict_single_image(model, weight_path, img_tensor)
print(f"Predicted Class: {label_map[predicted_class]}, Probability: {predicted_prob:.4f}")
