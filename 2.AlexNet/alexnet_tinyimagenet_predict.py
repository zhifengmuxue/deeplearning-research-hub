from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from AlexNet import AlexNet
import sys
import os
import random
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.predict_utils import predict_single_image

# AlexNet TinyImageNet 单图像预测脚本
# 先训练权重，在进行预测
# 定义类别映射
def get_tiny_imagenet_labels():
    """从TinyImageNet数据集目录自动生成类别映射"""
    class_dirs = os.listdir("./dataset/tiny-imagenet-200/train")
    class_to_idx = {}
    words_file = "./dataset/tiny-imagenet-200/words.txt"  # TinyImageNet包含此文件，映射ID到可读名称
    
    # 读取可读名称映射
    id_to_name = {}
    if os.path.exists(words_file):
        with open(words_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    id_to_name[parts[0]] = parts[1]
    
    # 创建类别到索引的映射
    for idx, class_id in enumerate(sorted(class_dirs)):
        if class_id != '.DS_Store':  # 忽略macOS系统文件
            readable_name = id_to_name.get(class_id, class_id)  # 如果找不到可读名称，使用ID
            class_to_idx[idx] = readable_name
    
    return class_to_idx
    

# 获取TinyImageNet标签映射
label_map = get_tiny_imagenet_labels()

# 定义transform（要与训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),     # AlexNet 要求大图像
    transforms.ToTensor()
])

# 加载训练集而不是验证集
try:
    test_dataset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=transform)
    print(f"成功加载TinyImageNet训练集，共有{len(test_dataset)}张图像")
except Exception as e:
    print(f"加载训练集失败: {e}")
    exit(1)

# 设置随机种子以便结果可复现
random.seed(11202)

# 随机获取图像和标签
random_idx = random.randint(0, len(test_dataset) - 1)
img_tensor, label = test_dataset[random_idx]

# 打印选择的索引，便于后续验证
print(f"随机选择的样本索引: {random_idx}")
print(f"真实标签索引: {label}")
print(f"真实标签名称: {label_map.get(label, '未知')}")

# 显示图像
img_to_show = img_tensor.permute(1, 2, 0).numpy()  # 调整维度顺序

plt.figure(figsize=(8, 8))
plt.imshow(img_to_show)
plt.title(f"True Label: {label_map.get(label, 'Unknown')}")
plt.axis("off")
plt.show()

# 初始化模型
model = AlexNet(input_shape=(3, 224, 224))

# 权重路径
current_dir = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(current_dir, "outputs", "weights", "AlexNet_TinyImageNet.pth")

# 检查权重文件是否存在
if not os.path.exists(weight_path):
    print(f"警告: 权重文件不存在: {weight_path}")
    print("请先运行训练脚本生成权重文件，或者检查权重文件路径是否正确")
    exit(1)

predicted_class, predicted_prob = predict_single_image(model, weight_path, img_tensor)
print(f"预测类别索引: {predicted_class}")
print(f"预测类别名称: {label_map.get(predicted_class, '未知')}")
print(f"预测置信度: {predicted_prob:.4f}")
    
# 获取前5个预测结果
model.load_state_dict(torch.load(weight_path))
model.eval()

with torch.no_grad():
    output = model(img_tensor.unsqueeze(0))
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    top5_prob, top5_indices = torch.topk(probabilities, 5)

print("\n前5个预测结果:")
for i in range(5):
    idx = top5_indices[i].item()
    prob = top5_prob[i].item()
    print(f"#{i+1}: {label_map.get(idx, '未知')} - {prob:.4f}")
    