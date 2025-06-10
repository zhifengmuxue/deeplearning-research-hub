from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from VGG import VGG16
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.predict_utils import predict_single_image


# 定义类别映射
label_map = {
    0: "飞机",
    1: "汽车",
    2: "鸟",
    3: "猫",
    4: "鹿",
    5: "狗",
    6: "青蛙",
    7: "马",
    8: "船",
    9: "卡车"
}
# 定义transform（要与训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),     # AlexNet 要求大图像
    transforms.Grayscale(num_output_channels=3),  # 1 通道变 3 通道
    transforms.ToTensor()
])
# 加载CIFAR-10测试集
test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
random.seed(88)

# 随机获取图像和标签
random_idx = random.randint(0, len(test_dataset) - 1)
img_tensor, label = test_dataset[random_idx]

# 打印选择的索引，便于后续验证
print(f"随机选择的样本索引: {random_idx}")

# 显示图像
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
img_to_show = img_tensor.permute(1, 2, 0)  # 将通道维度从(3,H,W)改为(H,W,3)
plt.imshow(img_to_show)
plt.title(f"True Label: {label_map[label]}")  # 使用 label_map 显示类别名称
plt.axis("off")
plt.show()

# 初始化模型
model = VGG16(input_shape=(3, 224, 224))

# 权重路径
current_dir = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(current_dir, "outputs", "weights", "AlexNet_CIFAR10.pth")

# 预测
predicted_class, predicted_prob = predict_single_image(model, weight_path, img_tensor)
print(f"Predicted Class: {predicted_class}, Probability: {predicted_prob:.4f}")

# 使用 label_map 显示预测类别名称
print(f"Predicted Class: {label_map[predicted_class]}, Probability: {predicted_prob:.4f}")
