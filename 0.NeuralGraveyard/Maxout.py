import torch
import torch.nn as nn
from torchinfo import summary as torchinfo_summary

# Maxout 网络被用来拟合任意的凸函数，
# 网络参数量提升了k倍，但是效果没有k倍的提升，已然被弃用了
# 这里仅简单复现，而不做更多的实验

class Maxout(nn.Module):
    def __init__(self, input_dim, output_dim, k=2):
        super(Maxout, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.linear = nn.Linear(input_dim, output_dim * k)

    def forward(self, x):
        *batch_dims, _ = x.shape
        x = self.linear(x)
        x = x.view(*batch_dims, self.output_dim, self.k)
        return torch.max(x, dim=-1)[0]

    def summary(self):
        return torchinfo_summary(
            self,
            input_size=(1, self.input_dim),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=2,
            device=torch.device("cpu")
        )
    
if __name__ == "__main__":
    model = Maxout(input_dim=10, output_dim=5, k=18)  # 5 * 18 
    print(model.summary())
