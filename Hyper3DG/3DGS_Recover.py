import torch

class GaussianRecovery:
    """3DGS恢复器：从patch级特征恢复到原始3DGS形状"""
    
    @staticmethod
    def recover_3DGS(representation, original_features, original_3DGS, patch_list, gs_dim):
        """
        从表示张量中恢复3DGS参数
        
        Args:
            representation (torch.Tensor): 表示张量 (X, X̃) [N, D]，包含3DGS特征和视觉特征
            original_features (torch.Tensor): 原始patch级特征 [N, D]
            original_3DGS: 原始3DGS参数
            patch_list (list): 包含N个patch的列表，每个patch包含该patch中的所有点
            gs_dim (int): 3DGS特征的维度（不包含视觉特征F的维度）
            
        Returns:
            dict: 更新后的3DGS参数
        """
        # 1. 删除视觉特征F，只保留3DGS特征 θ
        gs_features = representation[:, :gs_dim]  # [N, Cg]，N为patch数量
        original_gs_features = original_features[:, :gs_dim]  # [N, Cg]
        
        # 2. 计算patch级别的特征增量
        patch_deltas = gs_features - original_gs_features  # [N, Cg]
        
        # 3. 将特征增量应用到每个patch的所有点
        updated_3DGS = {
            '_xyz': original_3DGS._xyz.clone(),
            '_scaling': original_3DGS._scaling.clone(),
            '_rotation': original_3DGS._rotation.clone(),
            '_opacity': original_3DGS._opacity.clone(),
            '_features_dc': original_3DGS._features_dc.clone()
        }
        if hasattr(original_3DGS, '_features_rest') and original_3DGS._features_rest.numel() > 0:
            updated_3DGS['_features_rest'] = original_3DGS._features_rest.clone()
        
        current_idx = 0
        for patch_idx, patch in enumerate(patch_list):
            # 获取当前patch中点的数量
            num_points = len(patch._xyz)
            
            # 获取当前patch的特征增量
            start = 0
            
            # xyz增量
            xyz_delta = patch_deltas[patch_idx, start:start+3]
            updated_3DGS['_xyz'][current_idx:current_idx + num_points] += xyz_delta
            start += 3
            
            # scaling增量
            scaling_delta = patch_deltas[patch_idx, start:start+3]
            updated_3DGS['_scaling'][current_idx:current_idx + num_points] += scaling_delta
            start += 3
            
            # rotation增量
            rotation_delta = patch_deltas[patch_idx, start:start+4]
            updated_3DGS['_rotation'][current_idx:current_idx + num_points] += rotation_delta
            start += 4
            
            # opacity增量
            opacity_delta = patch_deltas[patch_idx, start:start+1]
            updated_3DGS['_opacity'][current_idx:current_idx + num_points] += opacity_delta
            start += 1
            
            # features_dc增量
            dc_dim = patch._features_dc.shape[1]
            dc_delta = patch_deltas[patch_idx, start:start+dc_dim]
            updated_3DGS['_features_dc'][current_idx:current_idx + num_points] += dc_delta
            start += dc_dim
            
            # features_rest增量（如果存在）
            if '_features_rest' in updated_3DGS:
                rest_dim = patch._features_rest.shape[1]
                rest_delta = patch_deltas[patch_idx, start:start+rest_dim]
                updated_3DGS['_features_rest'][current_idx:current_idx + num_points] += rest_delta
            
            current_idx += num_points
            
        return updated_3DGS 