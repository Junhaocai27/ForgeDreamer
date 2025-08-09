import torch
import numpy as np
from sklearn.cluster import KMeans
from scene import GaussianModel

class Patch3DGS:
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        
    def kmeans_clustering(self, positions, K):
        """
        使用K-means对3DGS点云进行聚类
        
        Args:
            positions (torch.Tensor): 3DGS点的3D坐标 [N, 3]
            K (int): 期望的patch数量
            
        Returns:
            list: 每个点属于哪个patch的索引列表 [points_num]
            numpy.ndarray: 聚类中心点 [K, 3]
        """
        positions_np = positions.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=K, random_state=42)
        kmeans.fit(positions_np)
        
        # 将labels转换为列表形式，每个元素表示对应点属于哪个patch (0 to K-1)
        point_to_patch = kmeans.labels_.tolist()
        cluster_centers = kmeans.cluster_centers_
        
        return point_to_patch, cluster_centers
    
    def get_patch_masks(self, patch_indices, K):
        """
        获取每个patch的布尔掩码
        
        Args:
            patch_indices (torch.Tensor): patch索引 [N]
            K (int): patch总数
            
        Returns:
            list[torch.Tensor]: K个布尔掩码的列表
        """
        return [patch_indices == i for i in range(K)]
    
    def split_3dgs(self, gaussians, K):
        """
        将3DGS通过K-means聚类重组
        
        Args:
            gaussians: GaussianModel实例
            K (int): 期望的patch数量
            
        Returns:
            tuple: (patch列表, 点到patch的映射列表, 聚类中心点)
        """
        # 获取高斯体位置并进行聚类
        positions = gaussians.get_xyz
        point_to_patch, cluster_centers = self.kmeans_clustering(positions, K)
        
        # 获取每个patch的掩码
        patch_masks = [torch.tensor(point_to_patch) == i for i in range(K)]
        
        # 按照聚类结果重组3DGS
        patch_gaussians = []
        for i, mask in enumerate(patch_masks):
            # 创建新的GaussianModel实例并保留属性
            patch_gaussian = GaussianModel(gaussians.active_sh_degree)
            # 复制原始高斯体的所有属性
            for attr in ['_xyz', '_features_dc', '_features_rest', 
                        '_scaling', '_rotation', '_opacity']:
                setattr(patch_gaussian, attr, getattr(gaussians, attr)[mask].clone())
            # 添加patch中心坐标
            patch_gaussian.patch_center = torch.from_numpy(cluster_centers[i]).to(positions.device)
            patch_gaussians.append(patch_gaussian)
            
        return patch_gaussians, point_to_patch, cluster_centers