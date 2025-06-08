import torch
# macos M1/M2 GPU support

if __name__ == "__main__":
    print(torch.backends.mps.is_available())       # True 表示支持Metal GPU
    print(torch.backends.mps.is_built())    