import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg import Hypergraph
from dhg.models import HGNNP

class HGNN_3D(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # 使用HGNNP替换HGNN
        self.hgnnp = HGNNP(
            in_channels=in_channels,
            hid_channels=hidden_channels,
            num_classes=in_channels,  # 输出维度与输入维度相同
            drop_rate=0.5
        )
        
        # 存储超图结构
        self.hypergraph = None
    
    def forward(self, x, hypergraph):
        """超图神经网络前向传播"""
            
        # 通过HGNNP处理特征
        refined_features = self.hgnnp(x, hypergraph)
        # 添加LeakyReLU激活函数
        refined_features = F.leaky_relu(refined_features, negative_slope=0.2)
        
        return refined_features
