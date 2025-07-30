import torch
import torch.nn.functional as F
import dhg
from dhg.nn import HGNNPConv
from einops import rearrange
import torch.nn as nn
import numpy as np

class DirectSimilarityDHGLatentHypergraph(nn.Module):
    """
    直接相似性计算版本 - 去除编码解码和分块计算
    先对denoised_latents和original_latents分别进行超图卷积，再计算差异
    """
    
    def __init__(self, device='cuda', reconstruction_interval=10, similarity_threshold=0.7, 
                 min_hyperedge_size=2, max_hyperedge_size=20):
        super().__init__()
        self.device = device
        self.reconstruction_interval = reconstruction_interval
        self.similarity_threshold = similarity_threshold
        self.min_hyperedge_size = min_hyperedge_size
        self.max_hyperedge_size = max_hyperedge_size
        self.last_reconstruction_iter = -1
        self.cached_original_hypergraph = None
        self.cached_denoised_hypergraph = None
        self.top_k = 16
        
        # 直接处理4通道特征
        self.hgnn_conv = HGNNPConv(4, 4).to(device)
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
    
    def should_reconstruct_hypergraph(self, iteration):
        """判断是否需要重构超图"""
        return (iteration - self.last_reconstruction_iter) >= self.reconstruction_interval
    
    def latents_to_node_features(self, latents):
        """
        将latents转换为节点特征矩阵
        Args:
            latents: [B, 4, 64, 64] 输入latents
        Returns:
            node_features: [B*64*64, 4] 节点特征矩阵
            batch_info: 用于重构的信息
        """
        B, C, H, W = latents.shape
        
        # 重塑为节点特征: [B, 4, 64, 64] -> [B*64*64, 4]
        node_features = latents.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        
        batch_info = {
            'batch_size': B,
            'channels': C,
            'height': H,
            'width': W,
            'total_nodes': B * H * W
        }
        
        return node_features, batch_info
    
    def node_features_to_latents(self, node_features, batch_info):
        """
        将节点特征转换回latents格式
        Args:
            node_features: [B*64*64, 4] 节点特征矩阵
            batch_info: 重构信息
        Returns:
            latents: [B, 4, 64, 64] 重构的latents
        """
        B = batch_info['batch_size']
        C = batch_info['channels']
        H = batch_info['height']
        W = batch_info['width']
        
        # 重塑回原始格式: [B*64*64, 4] -> [B, 4, 64, 64]
        latents = node_features.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return latents
    
    def create_topk_hypergraph(self, node_features, iteration, batch_info, hypergraph_type="original"):
        """
        使用Top-K方法构建超图。对于每个节点，选择与其最相似的K-1个节点共同构成一条超边。

        Args:
            node_features (torch.Tensor): [N, C] 节点特征矩阵, N = B*H*W。
            iteration (int): 当前迭代数。
            batch_info (dict): 包含 batch_size, height, width, total_nodes 的字典。
            hypergraph_type (str): "original" 或 "denoised"，用于区分和缓存。
        """
        # 1. --- 缓存逻辑 (与之前相同) ---
        # 根据类型选择缓存
        if hypergraph_type == "original":
            cached_hypergraph = self.cached_original_hypergraph
        else:
            cached_hypergraph = self.cached_denoised_hypergraph
            
        if not self.should_reconstruct_hypergraph(iteration) and cached_hypergraph is not None:
            return cached_hypergraph
        
        # print(f"[DEBUG] Creating Top-K hypergraph (k={self.top_k}) for {hypergraph_type} at iteration {iteration}")
        
        num_nodes = batch_info['total_nodes']
        B, H, W = batch_info['batch_size'], batch_info['height'], batch_info['width']
        
        # print(f"[DEBUG] Top-K computation for {num_nodes} nodes (B={B}, H={H}, W={W})")
        # print(f"[DEBUG] {hypergraph_type} node features shape: {node_features.shape}")
        
        hyperedge_list = []
        hyperedge_weights = []
        
        with torch.no_grad():
            # 2. --- 相似性矩阵计算 (与之前相同) ---
            # print(f"[DEBUG] Computing full similarity matrix for {hypergraph_type}...")
            normalized_features = F.normalize(node_features, p=2, dim=1)
            
            # 直接矩阵乘法 - 依然是内存消耗大户
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            # print(f"[DEBUG] {hypergraph_type} similarity matrix computed, shape: {similarity_matrix.shape}")
            
            # =====================================================================
            # 3. --- 【核心修改】基于Top-K构建超边 ---
            # =====================================================================
            # print(f"[DEBUG] Building hyperedges for {hypergraph_type} using Top-K (k={self.top_k})...")
            
            # 对角线元素（自身与自身的相似度）可能为1，会影响top-k结果，可以置为负无穷
            # 但通常包含自身作为超边的一部分是合理的，所以我们保留它
            
            # 一次性计算所有节点的Top-K邻居，效率更高
            # torch.topk返回两个张量: (top_values, top_indices)
            top_k_similarities, top_k_indices = torch.topk(
                similarity_matrix, 
                k=self.top_k, 
                dim=1, 
                largest=True
            )
            
            # print(f"[DEBUG] Top-K indices calculated, shape: {top_k_indices.shape}")

            # 将结果转换为超边列表和权重列表
            # 每一行 top_k_indices[i] 就是一条以节点i为中心的超边
            hyperedge_list = top_k_indices.tolist()
            
            # 使用这些top-k相似度的均值作为每条超边的权重
            hyperedge_weights = top_k_similarities.mean(dim=1).tolist()
            
            # print(f"[DEBUG] {hypergraph_type} initial hyperedges created: {len(hyperedge_list)}")
            # =====================================================================

            # 4. --- 去除重复的超边 (保持不变，依然是好的实践) ---
            # print(f"[DEBUG] Removing duplicate hyperedges for {hypergraph_type}...")
            unique_hyperedges = []
            unique_weights = []
            seen_edges = set()
            
            for edge, weight in zip(hyperedge_list, hyperedge_weights):
                # 将超边排序以创建唯一的标识符
                sorted_edge = tuple(sorted(edge))
                if sorted_edge not in seen_edges:
                    seen_edges.add(sorted_edge)
                    unique_hyperedges.append(edge)
                    unique_weights.append(weight)
            
            hyperedge_list = unique_hyperedges
            hyperedge_weights = unique_weights
            
            # print(f"[DEBUG] {hypergraph_type} final unique hyperedges: {len(hyperedge_list)}")
            
            if len(hyperedge_list) > 0:
                # 对于Top-K，平均大小理论上应接近K
                avg_edge_size = np.mean([len(edge) for edge in hyperedge_list])
                # print(f"[DEBUG] {hypergraph_type} average hyperedge size: {avg_edge_size:.2f}")

            # 5. --- 回退策略 (对于Top-K不再必要) ---
            # Top-K方法总是能为每个节点生成一条超边，所以几乎不可能出现 hyperedge_list 为空的情况。
            # 因此，可以安全地移除之前复杂的 fallback 逻辑，使代码更简洁。
            if len(hyperedge_list) == 0:
                print(f"[CRITICAL WARNING] Top-K method failed to produce any hyperedges. This is highly unexpected.")
                # 建立一个最小的连接图作为最后的手段
                hyperedge_list.append(list(range(self.top_k)))
                hyperedge_weights.append(0.5)

            # 6. --- 创建并缓存DHG超图 (与之前相同) ---
            # print(f"[DEBUG] Creating DHG hypergraph for {hypergraph_type}...")
            hypergraph = dhg.Hypergraph(num_nodes, hyperedge_list, device=self.device)
            
            hypergraph_data = {
                'hypergraph': hypergraph,
                'edge_weights': torch.tensor(hyperedge_weights, device=self.device),
                'num_nodes': num_nodes,
                'num_edges': len(hyperedge_list),
                'avg_edge_size': np.mean([len(edge) for edge in hyperedge_list]) if hyperedge_list else 0,
                'batch_info': batch_info,
                'k_used': self.top_k, # 记录使用的K值
                'type': hypergraph_type
            }
            
            # 根据类型缓存
            if hypergraph_type == "original":
                self.cached_original_hypergraph = hypergraph_data
            else:
                self.cached_denoised_hypergraph = hypergraph_data
            
            self.last_reconstruction_iter = iteration
            
            # print(f"[DEBUG] {hypergraph_type} Top-K hypergraph created successfully!")
            return hypergraph_data
    
    def create_direct_similarity_hypergraph(self, node_features, iteration, batch_info, hypergraph_type="original"):
        """
        直接计算相似性矩阵 - 不使用分块
        Args:
            node_features: [16384, 4] 节点特征矩阵
            iteration: 当前迭代数
            batch_info: batch信息
            hypergraph_type: "original" 或 "denoised"
        """
        # 根据类型选择缓存
        if hypergraph_type == "original":
            cached_hypergraph = self.cached_original_hypergraph
        else:
            cached_hypergraph = self.cached_denoised_hypergraph
            
        if not self.should_reconstruct_hypergraph(iteration) and cached_hypergraph is not None:
            return cached_hypergraph
        
        # print(f"[DEBUG] Computing direct similarity hypergraph for {hypergraph_type} at iteration {iteration}")
        
        num_nodes = batch_info['total_nodes']
        B, H, W = batch_info['batch_size'], batch_info['height'], batch_info['width']
        
        # print(f"[DEBUG] Direct computation for {num_nodes} nodes (B={B}, H={H}, W={W})")
        # print(f"[DEBUG] {hypergraph_type} node features shape: {node_features.shape}")
        
        hyperedge_list = []
        hyperedge_weights = []
        
        with torch.no_grad():
            # 直接计算全相似性矩阵 - 一次性计算，无分块
            # print(f"[DEBUG] Computing full similarity matrix for {hypergraph_type}...")
            normalized_features = F.normalize(node_features, p=2, dim=1)
            
            # 直接矩阵乘法 - 可能占用大量内存
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            # print(f"[DEBUG] {hypergraph_type} similarity matrix computed, shape: {similarity_matrix.shape}")
            # print(f"[DEBUG] Memory usage: {similarity_matrix.numel() * 4 / 1024 / 1024:.2f} MB")
            
            # 基于相似性阈值构建超边
            # print(f"[DEBUG] Building hyperedges for {hypergraph_type} with threshold {self.similarity_threshold}...")
            edges_created = 0
            
            for i in range(num_nodes):
                # 找到与节点i相似度超过阈值的所有节点
                similar_mask = similarity_matrix[i] > self.similarity_threshold
                similar_indices = torch.where(similar_mask)[0]
                
                # 过滤超边大小
                if len(similar_indices) >= self.min_hyperedge_size and len(similar_indices) <= self.max_hyperedge_size:
                    similar_nodes = similar_indices.tolist()
                    hyperedge_list.append(similar_nodes)
                    
                    # 使用相似性分数作为权重
                    similarities = similarity_matrix[i][similar_indices]
                    avg_similarity = similarities.mean().item()
                    hyperedge_weights.append(avg_similarity)
                    edges_created += 1
                
                # 打印进度
                if i % 2000 == 0:
                    progress = i / num_nodes * 100
                    # print(f"[DEBUG] {hypergraph_type} Progress: {progress:.1f}%, edges created: {edges_created}")
            
            # print(f"[DEBUG] {hypergraph_type} initial hyperedges created: {len(hyperedge_list)}")
            
            # 去除重复的超边
            # print(f"[DEBUG] Removing duplicate hyperedges for {hypergraph_type}...")
            unique_hyperedges = []
            unique_weights = []
            seen_edges = set()
            
            for edge, weight in zip(hyperedge_list, hyperedge_weights):
                sorted_edge = tuple(sorted(edge))
                if sorted_edge not in seen_edges:
                    seen_edges.add(sorted_edge)
                    unique_hyperedges.append(edge)
                    unique_weights.append(weight)
            
            hyperedge_list = unique_hyperedges
            hyperedge_weights = unique_weights
            
            # print(f"[DEBUG] {hypergraph_type} final unique hyperedges: {len(hyperedge_list)}")
            
            if len(hyperedge_list) > 0:
                avg_edge_size = np.mean([len(edge) for edge in hyperedge_list])
                # print(f"[DEBUG] {hypergraph_type} average hyperedge size: {avg_edge_size:.2f}")
            
            # 回退策略
            if len(hyperedge_list) == 0:
                print(f"[WARNING] No hyperedges created for {hypergraph_type}! Using fallback strategy.")
                # 降低阈值重试
                fallback_threshold = max(0.3, self.similarity_threshold - 0.2)
                print(f"[WARNING] Trying fallback threshold for {hypergraph_type}: {fallback_threshold}")
                
                for i in range(0, num_nodes, 100):
                    similar_mask = similarity_matrix[i] > fallback_threshold
                    similar_indices = torch.where(similar_mask)[0]
                    
                    if len(similar_indices) >= self.min_hyperedge_size:
                        similar_nodes = similar_indices.tolist()
                        hyperedge_list.append(similar_nodes)
                        hyperedge_weights.append(fallback_threshold)
                        
                        if len(hyperedge_list) >= 100:
                            break
            
            # 最小回退
            if len(hyperedge_list) == 0:
                print(f"[WARNING] Still no hyperedges for {hypergraph_type}! Creating minimal fallback.")
                for i in range(min(10, num_nodes // 1000)):
                    start_idx = i * 1000
                    end_idx = min(start_idx + 100, num_nodes)
                    if end_idx - start_idx >= self.min_hyperedge_size:
                        fallback_edge = list(range(start_idx, end_idx))
                        hyperedge_list.append(fallback_edge)
                        hyperedge_weights.append(0.5)
            
            # 创建DHG超图
            # print(f"[DEBUG] Creating DHG hypergraph for {hypergraph_type}...")
            hypergraph = dhg.Hypergraph(num_nodes, hyperedge_list, device=self.device)
            
            hypergraph_data = {
                'hypergraph': hypergraph,
                'edge_weights': torch.tensor(hyperedge_weights, device=self.device),
                'num_nodes': num_nodes,
                'num_edges': len(hyperedge_list),
                'avg_edge_size': np.mean([len(edge) for edge in hyperedge_list]) if hyperedge_list else 0,
                'batch_info': batch_info,
                'threshold_used': self.similarity_threshold,
                'type': hypergraph_type
            }
            
            # 根据类型缓存
            if hypergraph_type == "original":
                self.cached_original_hypergraph = hypergraph_data
            else:
                self.cached_denoised_hypergraph = hypergraph_data
            
            self.last_reconstruction_iter = iteration
            
            # print(f"[DEBUG] {hypergraph_type} hypergraph created successfully!")
            return hypergraph_data
    
    def apply_hypergraph_convolution(self, hypergraph_data, node_features):
        """应用超图卷积"""
        hypergraph = hypergraph_data['hypergraph']
        
        with torch.no_grad():
            original_dtype = node_features.dtype
            node_features_float = node_features.float()
            
            enhanced_features = self.hgnn_conv(node_features_float, hypergraph)
            enhanced_features = enhanced_features.to(original_dtype)
        
        return enhanced_features
    
    def transform_spatial_mask_to_node_mask(self, spatial_mask, node_features):
        """
        将一个空间掩码 [B, 1, H, W] 变换为节点掩码 [N, D]。
        """
        num_nodes, feature_dim = node_features.shape
        
        # 展平空间掩码以匹配节点顺序
        flattened_mask = spatial_mask.view(num_nodes)
        
        # 扩展以匹配特征维度
        node_mask = flattened_mask.unsqueeze(1).repeat(1, feature_dim)
        
        return node_mask
    
    def forward(self, original_latents, denoised_latents, iteration, target_grad_norm=None, mask1=None, mask2=None):
        """
        前向传播 - 先分别对两个latents进行超图卷积，再计算差异
        Args:
            original_latents: [B, 4, 64, 64] 原始latents
            denoised_latents: [B, 4, 64, 64] 去噪后的latents
            iteration: 当前迭代数
            target_grad_norm: 目标梯度幅度
        """
        original_dtype = original_latents.dtype
        
        with torch.no_grad():
            # 1. 转换为节点特征（无编码解码）
            original_latents_float = original_latents.float()
            denoised_latents_float = denoised_latents.float()
            
            original_node_features, batch_info = self.latents_to_node_features(original_latents_float)
            denoised_node_features, _ = self.latents_to_node_features(denoised_latents_float)
            
            # print(f"[DEBUG] Node features shape: {original_node_features.shape}")
            
            # 2. 分别构建两个超图
            original_hypergraph_data = self.create_topk_hypergraph(
                original_node_features, iteration, batch_info, hypergraph_type="original"
            )
            denoised_hypergraph_data = self.create_topk_hypergraph(
                denoised_node_features, iteration, batch_info, hypergraph_type="denoised"
            )
            
            # 3. 分别对两个latents进行超图卷积
            enhanced_original_features = self.apply_hypergraph_convolution(original_hypergraph_data, original_node_features)
            enhanced_denoised_features = self.apply_hypergraph_convolution(denoised_hypergraph_data, denoised_node_features)
            
            # 4. 计算增强后的差异
            # print(f"mask1 shape: {mask1.shape}")
            # print(f"mask2 shape: {mask2.shape}")
            # print(f"enhanced_denoised_features shape: {enhanced_denoised_features.shape}")
            # print(f"enhanced_original_features shape: {enhanced_original_features.shape}")

            node_mask_denoised = self.transform_spatial_mask_to_node_mask(mask1, enhanced_denoised_features)
            node_mask_original = self.transform_spatial_mask_to_node_mask(mask2, enhanced_original_features)
            
            enhanced_diff_features = (node_mask_denoised * enhanced_denoised_features) - (node_mask_original * enhanced_original_features)
            # enhanced_diff_features = enhanced_denoised_features - enhanced_original_features
            
            # 5. 转换回latents格式
            grad_enhancement = self.node_features_to_latents(enhanced_diff_features, batch_info)
            
            # 6. 计算损失 - 基于增强后的特征和原始特征
            # original_loss = F.mse_loss(enhanced_original_features, original_node_features)
            # denoised_loss = F.mse_loss(enhanced_denoised_features, denoised_node_features)
            # hypergraph_loss = (original_loss + denoised_loss) / 2
            
            hypergraph_loss = F.mse_loss(enhanced_denoised_features, enhanced_original_features)
            
            grad_enhancement = grad_enhancement.to(original_dtype)
        
        # 7. 记录统计信息
        if iteration % 50 == 0:
            with torch.no_grad():
                # 计算原始差异用于对比
                original_diff = denoised_latents_float - original_latents_float
                self.log_enhancement_statistics(grad_enhancement, original_hypergraph_data, denoised_hypergraph_data, iteration, original_diff)
        
        return grad_enhancement, hypergraph_loss
    
    def log_enhancement_statistics(self, grad_enhancement, original_hypergraph_data, denoised_hypergraph_data, iteration, original_diff):
        """记录增强统计信息"""
        with torch.no_grad():
            # 增强后的差异统计
            enhanced_diff_norm = torch.norm(grad_enhancement).item()
            enhanced_diff_mean = grad_enhancement.abs().mean().item()
            enhanced_diff_std = grad_enhancement.std().item()
            
            # 原始差异统计
            original_diff_norm = torch.norm(original_diff).item()
            original_diff_mean = original_diff.abs().mean().item()
            original_diff_std = original_diff.std().item()
            
            # 计算增强效果
            enhancement_ratio = enhanced_diff_norm / (original_diff_norm + 1e-8)
            
            # print(f"[DEBUG] Iteration {iteration}:")
            print(f"  - Total nodes: {original_hypergraph_data['num_nodes']}")
            print(f"  - Original hyperedges: {original_hypergraph_data['num_edges']}")
            print(f"  - Original avg edge size: {original_hypergraph_data['avg_edge_size']:.2f}")
            print(f"  - Denoised hyperedges: {denoised_hypergraph_data['num_edges']}")
            print(f"  - Denoised avg edge size: {denoised_hypergraph_data['avg_edge_size']:.2f}")
            print(f"  - Original diff norm: {original_diff_norm:.6f}")
            print(f"  - Original diff mean: {original_diff_mean:.6f}")
            print(f"  - Original diff std: {original_diff_std:.6f}")
            print(f"  - Enhanced diff norm: {enhanced_diff_norm:.6f}")
            print(f"  - Enhanced diff mean: {enhanced_diff_mean:.6f}")
            print(f"  - Enhanced diff std: {enhanced_diff_std:.6f}")
            print(f"  - Enhancement ratio: {enhancement_ratio:.3f}")
            # print(f"  - Similarity threshold: {original_hypergraph_data['threshold_used']}")

class SimplifiedDHGLatentHypergraph(nn.Module):
    """
    简化的DHG超图处理器 - 使用top-k相似性构建超边
    """
    
    def __init__(self, device='cuda', reconstruction_interval=10, top_k=16, min_hyperedge_size=2):
        super().__init__()
        self.device = device
        self.reconstruction_interval = reconstruction_interval
        self.top_k = top_k  # 每个节点选择top-k个最相似的节点
        self.min_hyperedge_size = min_hyperedge_size  # 最小超边大小
        self.last_reconstruction_iter = -1
        self.cached_hypergraph = None
        
        # 固定的编码器和解码器（无可学习参数）
        self.encoder_transform = nn.PixelUnshuffle(downscale_factor=2)
        self.decoder_transform = nn.PixelShuffle(upscale_factor=2)
        
        # 固定的超图卷积层（不更新参数）
        self.hgnn_conv = HGNNPConv(16, 16).to(device)
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
            
    def encode_latents(self, latents):
        """编码: [B, 4, 64, 64] -> [B, 16, 32, 32]"""
        return self.encoder_transform(latents)
    
    def decode_features(self, features):
        """解码: [B, 16, 32, 32] -> [B, 4, 64, 64]"""
        return self.decoder_transform(features)
    
    def should_reconstruct_hypergraph(self, iteration):
        """判断是否需要重构超图"""
        return (iteration - self.last_reconstruction_iter) >= self.reconstruction_interval
    
    def create_similarity_threshold_hypergraph_v2(self, node_features, iteration):
        """
        修改版本：使用相似性阈值替代top-k
        """
        if not self.should_reconstruct_hypergraph(iteration) and self.cached_hypergraph is not None:
            return self.cached_hypergraph
        
        # print(f"[DEBUG] Reconstructing similarity threshold hypergraph at iteration {iteration}")
        
        num_nodes, feature_dim = node_features.shape
        similarity_threshold = 0.7  # 可调参数
        
        hyperedge_list = []
        hyperedge_weights = []
        
        with torch.no_grad():
            # 计算节点特征的相似性矩阵
            normalized_features = F.normalize(node_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            # 使用相似性阈值而非top-k
            for i in range(num_nodes):
                # 找到相似度超过阈值的节点
                similar_mask = similarity_matrix[i] > similarity_threshold
                similar_indices = torch.where(similar_mask)[0]
                
                # 限制超边大小
                if len(similar_indices) >= 2 and len(similar_indices) <= 20:
                    similar_nodes = similar_indices.tolist()
                    hyperedge_list.append(similar_nodes)
                    
                    # 使用相似性分数作为权重
                    similarities = similarity_matrix[i][similar_indices]
                    avg_similarity = similarities.mean().item()
                    hyperedge_weights.append(avg_similarity)

        # 去除重复的超边（可选）
        unique_hyperedges = []
        unique_weights = []
        seen_edges = set()
        
        for edge, weight in zip(hyperedge_list, hyperedge_weights):
            # 将超边排序以便比较
            sorted_edge = tuple(sorted(edge))
            if sorted_edge not in seen_edges:
                seen_edges.add(sorted_edge)
                unique_hyperedges.append(edge)
                unique_weights.append(weight)
        
        hyperedge_list = unique_hyperedges
        hyperedge_weights = unique_weights
        
        # print(f"[DEBUG] Created {len(hyperedge_list)} top-k similarity hyperedges for {num_nodes} nodes")
        # print(f"[DEBUG] Average hyperedge size: {np.mean([len(edge) for edge in hyperedge_list]):.2f}")
        
        if len(hyperedge_list) == 0:
            print("[WARNING] No hyperedges created! Using identity transformation.")
            # 如果没有超边，创建一个包含所有节点的超边
            hyperedge_list = [list(range(num_nodes))]
            hyperedge_weights = [1.0]
        
        # 创建DHG超图
        hypergraph = dhg.Hypergraph(num_nodes, hyperedge_list, device=self.device)
        
        hypergraph_data = {
            'hypergraph': hypergraph,
            'edge_weights': torch.tensor(hyperedge_weights, device=self.device),
            'num_nodes': num_nodes,
            'feature_dim': feature_dim,
            'avg_edge_size': np.mean([len(edge) for edge in hyperedge_list]),
            'num_edges': len(hyperedge_list)
        }
        
        self.cached_hypergraph = hypergraph_data
        self.last_reconstruction_iter = iteration
        return hypergraph_data

    def create_topk_similarity_hypergraph(self, node_features, iteration):
        """创建基于top-k相似性的超图结构"""
        if not self.should_reconstruct_hypergraph(iteration) and self.cached_hypergraph is not None:
            return self.cached_hypergraph
        
        # print(f"[DEBUG] Reconstructing top-k similarity hypergraph at iteration {iteration}")
        
        num_nodes, feature_dim = node_features.shape
        # print(f"[DEBUG] Node features shape: {node_features.shape}, using top-k={self.top_k}")
        
        hyperedge_list = []
        hyperedge_weights = []
        
        # 基于top-k特征相似性构建超边
        with torch.no_grad():
            # 计算节点特征的相似性矩阵
            normalized_features = F.normalize(node_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            # 方法1: 为每个节点找到top-k相似的节点，构建超边
            for i in range(num_nodes):
                # 获取第i个节点与所有其他节点的相似度
                similarities = similarity_matrix[i]
                
                # 找到top-k个最相似的节点（包括自己）
                _, top_k_indices = torch.topk(similarities, min(self.top_k, num_nodes), largest=True)
                
                if len(top_k_indices) >= self.min_hyperedge_size:
                    similar_nodes = top_k_indices.tolist()
                    hyperedge_list.append(similar_nodes)
                    
                    # 使用top-k相似性分数作为权重
                    top_k_similarities = similarities[top_k_indices]
                    avg_similarity = top_k_similarities.mean().item()
                    hyperedge_weights.append(avg_similarity)
        
        # 去除重复的超边（可选）
        unique_hyperedges = []
        unique_weights = []
        seen_edges = set()
        
        for edge, weight in zip(hyperedge_list, hyperedge_weights):
            # 将超边排序以便比较
            sorted_edge = tuple(sorted(edge))
            if sorted_edge not in seen_edges:
                seen_edges.add(sorted_edge)
                unique_hyperedges.append(edge)
                unique_weights.append(weight)
        
        hyperedge_list = unique_hyperedges
        hyperedge_weights = unique_weights
        
        # print(f"[DEBUG] Created {len(hyperedge_list)} top-k similarity hyperedges for {num_nodes} nodes")
        # print(f"[DEBUG] Average hyperedge size: {np.mean([len(edge) for edge in hyperedge_list]):.2f}")
        
        if len(hyperedge_list) == 0:
            print("[WARNING] No hyperedges created! Using identity transformation.")
            # 如果没有超边，创建一个包含所有节点的超边
            hyperedge_list = [list(range(num_nodes))]
            hyperedge_weights = [1.0]
        
        # 创建DHG超图
        hypergraph = dhg.Hypergraph(num_nodes, hyperedge_list, device=self.device)
        
        hypergraph_data = {
            'hypergraph': hypergraph,
            'edge_weights': torch.tensor(hyperedge_weights, device=self.device),
            'num_nodes': num_nodes,
            'feature_dim': feature_dim,
            'avg_edge_size': np.mean([len(edge) for edge in hyperedge_list]),
            'num_edges': len(hyperedge_list)
        }
        
        self.cached_hypergraph = hypergraph_data
        self.last_reconstruction_iter = iteration
        return hypergraph_data
    
    def apply_hypergraph_convolution(self, hypergraph_data, node_features):
        """应用超图卷积"""
        hypergraph = hypergraph_data['hypergraph']
        
        with torch.no_grad():
            # 确保数据类型一致
            original_dtype = node_features.dtype
            node_features_float = node_features.float()
            
            # 应用超图卷积
            enhanced_features = self.hgnn_conv(node_features_float, hypergraph)
            
            # 转换回原始数据类型
            enhanced_features = enhanced_features.to(original_dtype)
        
        return enhanced_features
    
    def forward(self, original_latents, denoised_latents, iteration):
        """
        前向传播
        Args:
            original_latents: 原始latents [B, 4, 64, 64]
            denoised_latents: 去噪后的latents [B, 4, 64, 64]
            iteration: 当前迭代数
        """
        # 记录原始数据类型
        original_dtype = original_latents.dtype
        
        # 1. 编码latents到更高维度
        with torch.no_grad():
            original_latents_float = original_latents.float()
            denoised_latents_float = denoised_latents.float()
            
            original_encoded = self.encode_latents(original_latents_float)  # [B, 16, 32, 32]
            denoised_encoded = self.encode_latents(denoised_latents_float)  # [B, 16, 32, 32]
        
        # 2. 重塑为节点特征矩阵
        B, C, H, W = original_encoded.shape
        num_spatial_nodes = B * H * W  # 空间节点数量
        node_features_original = original_encoded.view(B, C, H * W).permute(0, 2, 1).contiguous().view(num_spatial_nodes, C)  # [B*H*W, 16]
        node_features_denoised = denoised_encoded.view(B, C, H * W).permute(0, 2, 1).contiguous().view(num_spatial_nodes, C)   # [B*H*W, 16]
        
        # print(f"[DEBUG] Node features shape: {node_features_original.shape}")
        
        # 3. 构建top-k相似性超图
        hypergraph_data = self.create_similarity_threshold_hypergraph_v2(node_features_original, iteration)
        
        # 4. 应用超图卷积
        original_enhanced = self.apply_hypergraph_convolution(hypergraph_data, node_features_original)
        denoised_enhanced = self.apply_hypergraph_convolution(hypergraph_data, node_features_denoised)
        
        # 5. 计算特征差异
        with torch.no_grad():
            feature_diff = original_enhanced - denoised_enhanced
            hypergraph_loss = F.mse_loss(original_enhanced, denoised_enhanced)
        
        # 6. 重塑回原始格式并解码
        feature_diff_reshaped = feature_diff.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, 16, 32, 32]
        
        with torch.no_grad():
            grad_enhancement = self.decode_features(feature_diff_reshaped)  # [B, 4, 64, 64]
            grad_enhancement = grad_enhancement.to(original_dtype)
        
        return grad_enhancement, hypergraph_loss

class ImprovedStaticDHGLatentHypergraph(nn.Module):
    """
    静态DHG超图处理器 - 修复数据类型不匹配问题
    """
    
    def __init__(self, device='cuda', reconstruction_interval=10):
        super().__init__()
        self.device = device
        self.reconstruction_interval = reconstruction_interval
        self.last_reconstruction_iter = -1
        self.cached_hypergraph = None
        
        # 固定的编码器和解码器（无可学习参数）
        self.encoder_transform = nn.PixelUnshuffle(downscale_factor=2)
        self.decoder_transform = nn.PixelShuffle(upscale_factor=2)
        
        # 固定的超图构建参数
        self.spatial_radius = 3
        self.similarity_threshold = 0.8
        self.cross_view_weight = 0.8
        
        # 固定的超图卷积层（不更新参数）
        self.hgnn_conv = HGNNPConv(16, 16).to(device)
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
            
    def encode_latents(self, latents):
        """优化的固定编码: [B, 4, 64, 64] -> [B, 16, 32, 32]"""
        return self.encoder_transform(latents)
    
    def decode_features(self, features):
        """优化的固定解码: [B, 16, 32, 32] -> [B, 4, 64, 64]"""
        return self.decoder_transform(features)
    
    def should_reconstruct_hypergraph(self, iteration):
        """判断是否需要重构超图"""
        return (iteration - self.last_reconstruction_iter) >= self.reconstruction_interval
    
    def create_hypergraph_structure(self, encoded_features, iteration):
        """创建DHG超图结构（定时重构）"""
        if not self.should_reconstruct_hypergraph(iteration) and self.cached_hypergraph is not None:
            return self.cached_hypergraph
        
        # print(f"[DEBUG] Reconstructing hypergraph at iteration {iteration}")
        
        B, C, H, W = encoded_features.shape
        num_nodes = B * H * W
        
        node_features = encoded_features.view(B, C, H*W).permute(0, 2, 1).contiguous().view(num_nodes, C)
        
        hyperedge_list = []
        hyperedge_weights = []
        
        # 1. 空间邻域超边
        for b in range(B):
            batch_offset = b * H * W
            step = max(1, self.spatial_radius)
            for i in range(0, H, step):
                for j in range(0, W, step):
                    spatial_edge = []
                    for di in range(-self.spatial_radius, self.spatial_radius + 1, step):
                        for dj in range(-self.spatial_radius, self.spatial_radius + 1, step):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < H and 0 <= nj < W:
                                spatial_edge.append(batch_offset + ni * W + nj)
                    if len(spatial_edge) > 1:
                        hyperedge_list.append(spatial_edge)
                        hyperedge_weights.append(1.0)
        
        # 2. 特征相似性超边 (简化版)
        with torch.no_grad():
            sample_nodes = min(num_nodes, 512)
            node_indices = torch.randperm(num_nodes)[:sample_nodes]
            sampled_features = node_features[node_indices]
            normalized_features = F.normalize(sampled_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            for i, node_idx in enumerate(node_indices):
                similar_indices = torch.where(similarity_matrix[i] > self.similarity_threshold)[0]
                if len(similar_indices) > 1:
                    similar_nodes = [node_indices[idx].item() for idx in similar_indices]
                    hyperedge_list.append(similar_nodes)
                    hyperedge_weights.append(similarity_matrix[i][similar_indices].mean().item())
        
        # 3. 跨视图连接超边
        if B > 1:
            step = max(1, H // 8)
            for i in range(0, H, step):
                for j in range(0, W, step):
                    cross_view_edge = [b * H * W + i * W + j for b in range(B)]
                    hyperedge_list.append(cross_view_edge)
                    hyperedge_weights.append(self.cross_view_weight)
        
        # print(f"[DEBUG] Created {len(hyperedge_list)} hyperedges for {num_nodes} nodes")
        
        hypergraph = dhg.Hypergraph(num_nodes, hyperedge_list, device=self.device)
        
        hypergraph_data = {
            'hypergraph': hypergraph,
            'node_features': node_features,
            'edge_weights': torch.tensor(hyperedge_weights, device=self.device),
            'shape_info': (B, C, H, W)
        }
        
        self.cached_hypergraph = hypergraph_data
        self.last_reconstruction_iter = iteration
        return hypergraph_data
    
    def apply_static_hypergraph_processing(self, hypergraph_data, input_features):
        """应用静态超图处理（无参数更新）- 修复数据类型问题"""
        hypergraph = hypergraph_data['hypergraph']
        B, C, H, W = hypergraph_data['shape_info']
        
        node_features = input_features.view(B, C, H*W).permute(0, 2, 1).contiguous().view(B*H*W, C)
        
        with torch.no_grad():
            # 确保数据类型一致 - 转换为float32进行计算
            original_dtype = node_features.dtype
            node_features_float = node_features.float()
            
            enhanced_features = self.hgnn_conv(node_features_float, hypergraph)
            
            # 转换回原始数据类型
            enhanced_features = enhanced_features.to(original_dtype)
        
        enhanced_features = enhanced_features.view(B, H, W, C).permute(0, 3, 1, 2)
        return enhanced_features
    
    def forward(self, original_latents, denoised_latents, iteration):
        """
        前向传播 - 静态处理，无参数学习
        Args:
            original_latents: 原始latents [B, 4, 64, 64]
            denoised_latents: 去噪后的latents (latents_noisy - pred_noise) [B, 4, 64, 64]
            iteration: 当前迭代数
        """
        # 记录原始数据类型
        original_dtype = original_latents.dtype
        
        # 1. 编码latents (使用PixelUnshuffle) - 确保使用float32进行计算
        with torch.no_grad():
            # 转换为float32进行编码
            original_latents_float = original_latents.float()
            denoised_latents_float = denoised_latents.float()
            
            original_encoded = self.encode_latents(original_latents_float)
            denoised_encoded = self.encode_latents(denoised_latents_float)
        
        # 2. 构建或获取缓存的超图结构
        hypergraph_data = self.create_hypergraph_structure(original_encoded, iteration)
        
        # 3. 应用静态超图处理
        original_enhanced = self.apply_static_hypergraph_processing(hypergraph_data, original_encoded)
        denoised_enhanced = self.apply_static_hypergraph_processing(hypergraph_data, denoised_encoded)
        
        # 4. 计算特征差异
        with torch.no_grad():
            # 使用去噪后的latent与原始latent的差异
            feature_diff = original_enhanced - denoised_enhanced
            
            # 计算MSE损失（原始latent vs 去噪latent）
            hypergraph_loss = F.mse_loss(original_enhanced, denoised_enhanced)
        
        # 5. 解码回原始尺寸 (使用PixelShuffle)
        with torch.no_grad():
            grad_enhancement = self.decode_features(feature_diff)
            
            # 转换回原始数据类型
            grad_enhancement = grad_enhancement.to(original_dtype)
        
        return grad_enhancement, hypergraph_loss

class StaticGradientHypergraphEnhancer:
    """
    一个静态的、非训练的超图梯度增强器。
    它通过在超边定义的节点集内对梯度patch进行平均来平滑和结构化梯度。
    """
    def __init__(self, patch_size=4, alpha=0.5, similarity_threshold=0.8, device='cuda'):
        self.patch_size = patch_size
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # 由于是非训练模块，不需要任何可学习参数
        print(f"[INFO] Initialized StaticGradientHypergraphEnhancer with patch_size={patch_size}, alpha={alpha}")

    def build_and_average(self, patches: torch.Tensor, num_patches_h, num_patches_w):
        """
        构建超图并直接在超边上进行特征平均。
        Args:
            patches (torch.Tensor): [N, P_dim] 单个样本的梯度patch (N = H*W, P_dim = C*p*p)
        Returns:
            torch.Tensor: [N, P_dim] 增强后的梯度patch
        """
        N, P_dim = patches.shape
        hyperedges = []

        # 1. 基于特征相似性构建超边
        # 使用L2范数归一化来计算余弦相似度
        norm_patches = F.normalize(patches, p=2, dim=1)
        sim_matrix = torch.matmul(norm_patches, norm_patches.t())

        for i in range(N):
            similar_nodes = torch.where(sim_matrix[i] > self.similarity_threshold)[0]
            if len(similar_nodes) > 1:
                hyperedges.append(similar_nodes.tolist())

        # 2. 基于空间邻近性构建超边
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                neighbors = []
                # 3x3 邻域
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < num_patches_h and 0 <= nj < num_patches_w:
                            neighbors.append(ni * num_patches_w + nj)
                if len(neighbors) > 1:
                    hyperedges.append(neighbors)
        
        if not hyperedges: # 如果没有超边，直接返回原始patch
            return patches

        # 3. 在超边上进行平均 (模拟超图卷积)
        # 初始化一个与原始patch形状相同的零张量来累加结果
        enhanced_patches = torch.zeros_like(patches)
        # 创建一个计数器来记录每个patch被多少个超边覆盖
        counts = torch.zeros(N, 1, device=self.device)

        unique_hyperedges = list(set(map(tuple, hyperedges))) # 去重

        for edge_nodes in unique_hyperedges:
            edge_nodes_tensor = torch.tensor(edge_nodes, device=self.device, dtype=torch.long)
            
            # 计算这条超边上所有patch的平均值
            mean_patch = torch.mean(patches[edge_nodes_tensor], dim=0)
            
            # 将这个平均值加到所有属于该超边的patch上
            enhanced_patches[edge_nodes_tensor] += mean_patch
            counts[edge_nodes_tensor] += 1
        
        # 防止除以零，对于没有被任何超边覆盖的节点，保持原样
        counts[counts == 0] = 1
        
        # 计算加权平均
        averaged_patches = enhanced_patches / counts
        
        # 对于没有被任何超边覆盖的节点，使用其原始值
        uncovered_nodes_mask = (counts.squeeze() == 1) # counts初始化为0，只在被覆盖时增加
        # 注意：这里有一个逻辑细节，如果一个节点只被一个超边覆盖，它也会被平均。
        # 我们应该找到那些一个超边都没覆盖的节点。
        # 一个更鲁棒的方法是：
        final_patches = torch.where(counts > 1, averaged_patches, patches)
        
        return final_patches

    @torch.no_grad() # 明确表示此函数不应追踪梯度
    def __call__(self, grad_latent: torch.Tensor):
        B, C, H, W = grad_latent.shape
        p = self.patch_size
        num_patches_h, num_patches_w = H // p, W // p

        # 1. 将梯度转换为patch表示
        patches = rearrange(grad_latent, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        
        enhanced_patches_batch = []
        for i in range(B): # 对batch中的每个样本单独处理
            single_patches = patches[i] # [N, P_dim]
            enhanced_single_patches = self.build_and_average(single_patches, num_patches_h, num_patches_w)
            enhanced_patches_batch.append(enhanced_single_patches)
        
        enhanced_patches = torch.stack(enhanced_patches_batch, dim=0) # [B, N, P_dim]

        # 2. 重构梯度
        reconstructed_grad = rearrange(
            enhanced_patches, 
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=num_patches_h, w=num_patches_w, p1=p, p2=p, c=C
        )

        # 3. 残差连接
        final_grad = self.alpha * reconstructed_grad + (1 - self.alpha) * grad_latent
        
        # 4. 稳定化 (可选但推荐)
        orig_norm = torch.norm(grad_latent, p=2, dim=(1,2,3), keepdim=True)
        final_norm = torch.norm(final_grad, p=2, dim=(1,2,3), keepdim=True)
        
        scale = torch.clamp(orig_norm / (final_norm + 1e-9), min=0.5, max=1.5)
        final_grad = final_grad * scale
        
        return final_grad