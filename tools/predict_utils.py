import torch
import torch.nn.functional as F

def predict_single_image(model, weight_path, image_tensor, device=None):
    """
    加载模型权重，预测单张图像的类别。

    参数:
    - model: 你的模型实例（比如 LeNet5）
    - weight_path: 训练好的权重文件路径（.pth）
    - image_tensor: 输入图像张量，形状应是 (C, H, W)，未batch，且已经做过归一化等预处理
    - device: 设备，默认自动选择 CUDA/CPU

    返回:
    - predicted_class: 预测类别的整数标签
    - predicted_prob: 对应的概率（最大softmax值）
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # 给图片加batch维度，变成(1, C, H, W)
        input_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(input_tensor)  # (1, num_classes)
        probs = F.softmax(output, dim=1)
        predicted_prob, predicted_class = torch.max(probs, dim=1)

    return predicted_class.item(), predicted_prob.item()
