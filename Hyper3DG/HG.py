import torch
import torch.nn.functional as F
from dhg import Hypergraph

class HypergraphBuilder:
    def __init__(self, k_neighbors):
        """初始化超图构建器
        
        Args:
            k_neighbors (int): KNN中的邻居数量
        """
        self.k_neighbors = k_neighbors
        self.hypergraph = None

    def build_hypergraph(self, features):
        """使用DHG内置kNN方法构建超图
        
        Args:
            features (torch.Tensor): 节点特征矩阵 [N, feature_dim]
        """
        N = features.size(0)
        device = features.device
        
        # 创建空超图
        self.hypergraph = Hypergraph(num_v=N, device=device)
        
        # 使用内置kNN方法添加超边
        self.hypergraph.add_hyperedges_from_feature_kNN(
            features,  # 特征矩阵
            k=self.k_neighbors + 1  # 包含自身节点
        )
        
        return self.hypergraph

    def get_feature_updates(self, x_refined, x_original):
        """计算特征更新量
        
        Args:
            x_refined (torch.Tensor): 优化后的特征
            x_original (torch.Tensor): 原始特征
            
        Returns:
            torch.Tensor: 特征更新量
        """
        return x_refined - x_original
    
    def get_H_matrix(self):
        """获取超图的邻接矩阵
        
        Returns:
            torch.Tensor: 超图邻接矩阵
        """
        return self.hypergraph.H if self.hypergraph is not None else None
    
    def write_H_matrix(self, features, H_matrix):
        """写入超图的邻接矩阵
        
        Args:
            features (torch.Tensor): 节点特征矩阵
            H_matrix (torch.Tensor): 要写入的邻接矩阵
        """
        self.hypergraph = Hypergraph(
            num_v=features.size(0),
            e_list=H_matrix,
            e_weight=1.0,
        )