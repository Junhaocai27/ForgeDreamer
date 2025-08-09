import torch
from scene import GaussianModel
import copy

class GaussianRecovery:
    """3DGS恢复器：从patch级特征恢复到原始3DGS形状"""
    
    @staticmethod
    def recover_3DGS(updated_features, original_features, original_3DGS, point_to_patch, gs_dim, k_means_num):
        """
        从表示张量中恢复3DGS参数
        
        Args:
            updated_features (torch.Tensor): 更新后的特征 [N, D]
            original_features (torch.Tensor): 原始特征 [N, D]
            original_3DGS: 原始3DGS参数
            point_to_patch: 点到patch的映射
            gs_dim (int): 3DGS特征的维度
            k_means_num (int): patch数量
        """
        # 计算更新量
        update_features = updated_features[:, :gs_dim] - original_features
        
        # 初始化更新后的3DGS参数
        xyz_update = original_3DGS._xyz.clone()
        scaling_update = original_3DGS._scaling.clone()
        rotation_update = original_3DGS._rotation.clone()
        opacity_update = original_3DGS._opacity.clone()
        dc_update = original_3DGS._features_dc.clone()
        
        # 获取dc特征的维度
        dc_dim = original_3DGS._features_dc.shape[-1]
        
        alpha = 0.1  # 更新比例，根据需求调整

        # 遍历每个点，直接用 patch_idx 索引 update_features
        for point_idx, patch_idx in enumerate(point_to_patch):
            # 如果 patch_idx 超出 k_means_num 范围，就跳过
            if not (0 <= patch_idx < k_means_num):
                continue
            
            # 判断 update_features 的列数是否满足要求
            if update_features.shape[1] >= 11 + dc_dim:
                # 取出该 patch_idx 对应的一行数据
                patch_data = update_features[patch_idx]
                
                xyz_update[point_idx] = (1 - alpha) * original_3DGS._xyz[point_idx] + alpha * patch_data[:3]
                scaling_update[point_idx] = (1 - alpha) * original_3DGS._scaling[point_idx] + alpha * patch_data[3:6]
                rotation_update[point_idx] = (1 - alpha) * original_3DGS._rotation[point_idx] + alpha * patch_data[6:10]
                opacity_update[point_idx] = (1 - alpha) * original_3DGS._opacity[point_idx] + alpha * patch_data[10:11]
                dc_update[point_idx] = (1 - alpha) * original_3DGS._features_dc[point_idx] + alpha * patch_data[11 : 11 + dc_dim]

        updated_3DGS = {
            '_xyz': xyz_update,
            '_scaling': scaling_update,
            '_rotation': rotation_update,
            '_opacity': opacity_update,
            '_features_dc': dc_update,
            '_features_rest': original_3DGS._features_rest
        }
        
        # 如果存在rest特征，则更新
        if hasattr(original_3DGS, '_features_rest') and original_3DGS._features_rest.numel() > 0:
            rest_update = original_3DGS._features_rest.clone()
            rest_dim = original_3DGS._features_rest.shape[-1]
            if update_features.shape[1] >= 11 + dc_dim + rest_dim:
                for point_idx in range(len(point_to_patch)):
                    patch_idx = point_to_patch[point_idx]
                    for i in range(k_means_num):
                        if patch_idx == i:
                            rest_update[point_idx] += update_features[i, 11+dc_dim:11+dc_dim+rest_dim]
            updated_3DGS['_features_rest'] = rest_update
        
        return updated_3DGS

    @staticmethod
    def get_feature_dims(original_3DGS):
        """
        获取3DGS各个特征的维度
        
        参数:
            original_3DGS: 原始的3DGS参数对象
            
        返回:
            dict: 包含各个特征维度的字典
        """
        feature_dims = {
            'xyz': 3,
            'scaling': 3,
            'rotation': 4,
            'opacity': 1,
            'features': original_3DGS._features_dc.shape[-1] + 
                      (original_3DGS._features_rest.shape[-1] if original_3DGS._features_rest.numel() > 0 else 0)
        }
        return feature_dims

    @staticmethod
    def update_gaussians(gaussians, updated_3DGS):
        """
        在原有高斯体对象上更新 3DGS 参数，而不新建对象
        
        参数:
            gaussians (GaussianModel): 需要更新的原始高斯体对象
            updated_3DGS (dict): 更新后的 3DGS 参数字典
        
        返回:
            gaussians: 更新后的高斯体对象（原地修改）
        """
        # 直接更新每个属性，确保完整覆盖
        if '_xyz' in updated_3DGS:
            gaussians._xyz = updated_3DGS['_xyz'].clone()  # 避免共享引用
        if '_scaling' in updated_3DGS:
            gaussians._scaling = updated_3DGS['_scaling'].clone()
        if '_rotation' in updated_3DGS:
            gaussians._rotation = updated_3DGS['_rotation'].clone()
        if '_opacity' in updated_3DGS:
            gaussians._opacity = updated_3DGS['_opacity'].clone()
        if '_features_dc' in updated_3DGS:
            gaussians._features_dc = updated_3DGS['_features_dc'].clone()
        
        # `features_rest` 可能为空，需要手动判断
        if '_features_rest' in updated_3DGS:
            gaussians._features_rest = updated_3DGS['_features_rest'].clone() if updated_3DGS['_features_rest'] is not None else None
        
        return gaussians  # 返回更新后的对象
