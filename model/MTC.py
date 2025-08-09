import torch.nn as nn

# 映射到 CLIP 空间
class MapToClip(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MapToClip, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)