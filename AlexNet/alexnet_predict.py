from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from AlexNet import AlexNet
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.predict_utils import predict_single_image

### AlexNet MNIST 单图像预测脚本
### 先训练权重，在进行预测

# 定义transform（要与训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),     # AlexNet 要求大图像
    transforms.Grayscale(num_output_channels=3),  # 1 通道变 3 通道
    transforms.ToTensor()
])

# 加载MNIST测试集
test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

random.seed(42)

# 随机获取图像和标签
random_idx = random.randint(0, len(test_dataset) - 1)
img_tensor, label = test_dataset[random_idx]

# 打印选择的索引，便于后续验证
print(f"随机选择的样本索引: {random_idx}")

# 显示图像
img_to_show = img_tensor.permute(1, 2, 0)  # 将通道维度从(3,H,W)改为(H,W,3)

plt.imshow(img_to_show)  
plt.title(f"True Label: {label}")
plt.axis("off")
plt.show()

# 初始化模型
model = AlexNet(input_shape=(3, 224, 224))

# 权重路径
current_dir = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(current_dir, "outputs", "weights", "AlexNet_MNIST.pth")

# 预测
predicted_class, predicted_prob = predict_single_image(model, weight_path, img_tensor)
print(f"Predicted Class: {predicted_class}, Probability: {predicted_prob:.4f}")
