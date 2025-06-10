from VGG import VGG_A, VGG_B, VGG_C, VGG16, VGG19, VGG_A_LRN
from torchinfo import summary as torchinfo_summary
import torch
import argparse

"""
# 显示 VGG-A 结构
python netwrok_visual.py --model a

# 显示 VGG-B 结构
python netwrok_visual.py --model b

# 或者使用简短形式
python netwrok_visual.py -m c
"""

def show_model_summary(model_name):
    # 根据输入字符串选择相应的 VGG 模型
    models = {
        'a': VGG_A,
        'a-lrn': VGG_A_LRN,
        'b': VGG_B,
        'c': VGG_C,
        'd': VGG16,
        'e': VGG19
    }
    
    # 检查模型名称是否有效
    if model_name.lower() not in models:
        print(f"错误: 未知的模型 '{model_name}'")
        print(f"可用模型: {', '.join(models.keys())}")
        return
    
    # 创建模型实例
    model_class = models[model_name.lower()]
    model = model_class(input_shape=(3, 224, 224), num_classes=1000)
    
    # 显示模型结构摘要
    print(f"\n{'='*50}")
    print(f"VGG-{model_name.upper()} 模型架构")
    print(f"{'='*50}")
    
    # 使用 torchinfo 提供详细的模型摘要
    summary = torchinfo_summary(
        model, 
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=3,
        device=torch.device("cpu")
    )
    
    # 输出一些额外的模型统计信息
    print(f"\n总参数量: {summary.total_params:,}")
    print(f"可训练参数量: {summary.trainable_params:,}")
    print(f"推理所需 FLOPs: {summary.total_mult_adds:,}")

if __name__ == "__main__":
    # 使用命令行参数选择模型
    parser = argparse.ArgumentParser(description='VGG 模型结构可视化')
    parser.add_argument('--model', '-m', type=str, default='a', 
                        help='VGG模型变体 (a/b/c/d/e)')
    args = parser.parse_args()
    
    show_model_summary(args.model)
    
    # 如果要展示所有模型，取消下面的注释
    # for model_name in ['a', 'a-lrn' ,'b', 'c', 'd', 'e']:
    #     show_model_summary(model_name)