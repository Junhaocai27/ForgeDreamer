import torch
import torch.nn.functional as F
import dhg
from dhg.nn import HGNNPConv
from einops import rearrange
import torch.nn as nn
import numpy as np

class DirectSimilarityDHGLatentHypergraph(nn.Module):
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
        
        self.hgnn_conv = HGNNPConv(4, 4).to(device)
        
        for param in self.parameters():
            param.requires_grad = False
    
    def should_reconstruct_hypergraph(self, iteration):
        return (iteration - self.last_reconstruction_iter) >= self.reconstruction_interval
    
    def latents_to_node_features(self, latents):
        B, C, H, W = latents.shape
        
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
        B = batch_info['batch_size']
        C = batch_info['channels']
        H = batch_info['height']
        W = batch_info['width']
        
        latents = node_features.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return latents
    
    def create_topk_hypergraph(self, node_features, iteration, batch_info, hypergraph_type="original"):
        if hypergraph_type == "original":
            cached_hypergraph = self.cached_original_hypergraph
        else:
            cached_hypergraph = self.cached_denoised_hypergraph
            
        if not self.should_reconstruct_hypergraph(iteration) and cached_hypergraph is not None:
            return cached_hypergraph
        
        num_nodes = batch_info['total_nodes']
        B, H, W = batch_info['batch_size'], batch_info['height'], batch_info['width']
        
        hyperedge_list = []
        hyperedge_weights = []
        
        with torch.no_grad():
            normalized_features = F.normalize(node_features, p=2, dim=1)
            
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            top_k_similarities, top_k_indices = torch.topk(
                similarity_matrix, 
                k=self.top_k, 
                dim=1, 
                largest=True
            )
            
            hyperedge_list = top_k_indices.tolist()
            
            hyperedge_weights = top_k_similarities.mean(dim=1).tolist()
            
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
            
            if len(hyperedge_list) == 0:
                print(f"[CRITICAL WARNING] Top-K method failed to produce any hyperedges. This is highly unexpected.")
                hyperedge_list.append(list(range(self.top_k)))
                hyperedge_weights.append(0.5)

            hypergraph = dhg.Hypergraph(num_nodes, hyperedge_list, device=self.device)
            
            hypergraph_data = {
                'hypergraph': hypergraph,
                'edge_weights': torch.tensor(hyperedge_weights, device=self.device),
                'num_nodes': num_nodes,
                'num_edges': len(hyperedge_list),
                'avg_edge_size': np.mean([len(edge) for edge in hyperedge_list]) if hyperedge_list else 0,
                'batch_info': batch_info,
                'k_used': self.top_k,
                'type': hypergraph_type
            }
            
            if hypergraph_type == "original":
                self.cached_original_hypergraph = hypergraph_data
            else:
                self.cached_denoised_hypergraph = hypergraph_data
            
            self.last_reconstruction_iter = iteration
            
            return hypergraph_data
    
    def create_direct_similarity_hypergraph(self, node_features, iteration, batch_info, hypergraph_type="original"):
        if hypergraph_type == "original":
            cached_hypergraph = self.cached_original_hypergraph
        else:
            cached_hypergraph = self.cached_denoised_hypergraph
            
        if not self.should_reconstruct_hypergraph(iteration) and cached_hypergraph is not None:
            return cached_hypergraph
        
        num_nodes = batch_info['total_nodes']
        B, H, W = batch_info['batch_size'], batch_info['height'], batch_info['width']
        
        hyperedge_list = []
        hyperedge_weights = []
        
        with torch.no_grad():
            normalized_features = F.normalize(node_features, p=2, dim=1)
            
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            edges_created = 0
            
            for i in range(num_nodes):
                similar_mask = similarity_matrix[i] > self.similarity_threshold
                similar_indices = torch.where(similar_mask)[0]
                
                if len(similar_indices) >= self.min_hyperedge_size and len(similar_indices) <= self.max_hyperedge_size:
                    similar_nodes = similar_indices.tolist()
                    hyperedge_list.append(similar_nodes)
                    
                    similarities = similarity_matrix[i][similar_indices]
                    avg_similarity = similarities.mean().item()
                    hyperedge_weights.append(avg_similarity)
                    edges_created += 1
                
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
            
            if len(hyperedge_list) == 0:
                print(f"[WARNING] No hyperedges created for {hypergraph_type}! Using fallback strategy.")
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
            
            if len(hyperedge_list) == 0:
                print(f"[WARNING] Still no hyperedges for {hypergraph_type}! Creating minimal fallback.")
                for i in range(min(10, num_nodes // 1000)):
                    start_idx = i * 1000
                    end_idx = min(start_idx + 100, num_nodes)
                    if end_idx - start_idx >= self.min_hyperedge_size:
                        fallback_edge = list(range(start_idx, end_idx))
                        hyperedge_list.append(fallback_edge)
                        hyperedge_weights.append(0.5)
            
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
            
            if hypergraph_type == "original":
                self.cached_original_hypergraph = hypergraph_data
            else:
                self.cached_denoised_hypergraph = hypergraph_data
            
            self.last_reconstruction_iter = iteration
            
            return hypergraph_data
    
    def apply_hypergraph_convolution(self, hypergraph_data, node_features):
        hypergraph = hypergraph_data['hypergraph']
        
        with torch.no_grad():
            original_dtype = node_features.dtype
            node_features_float = node_features.float()
            
            enhanced_features = self.hgnn_conv(node_features_float, hypergraph)
            enhanced_features = enhanced_features.to(original_dtype)
        
        return enhanced_features
    
    def transform_spatial_mask_to_node_mask(self, spatial_mask, node_features):
        num_nodes, feature_dim = node_features.shape
        
        flattened_mask = spatial_mask.view(num_nodes)
        
        node_mask = flattened_mask.unsqueeze(1).repeat(1, feature_dim)
        
        return node_mask
    
    def forward(self, original_latents, denoised_latents, iteration, target_grad_norm=None, mask1=None, mask2=None):
        original_dtype = original_latents.dtype
        
        with torch.no_grad():
            original_latents_float = original_latents.float()
            denoised_latents_float = denoised_latents.float()
            
            original_node_features, batch_info = self.latents_to_node_features(original_latents_float)
            denoised_node_features, _ = self.latents_to_node_features(denoised_latents_float)
            
            original_hypergraph_data = self.create_topk_hypergraph(
                original_node_features, iteration, batch_info, hypergraph_type="original"
            )
            denoised_hypergraph_data = self.create_topk_hypergraph(
                denoised_node_features, iteration, batch_info, hypergraph_type="denoised"
            )
            
            enhanced_original_features = self.apply_hypergraph_convolution(original_hypergraph_data, original_node_features)
            enhanced_denoised_features = self.apply_hypergraph_convolution(denoised_hypergraph_data, denoised_node_features)
            
            node_mask_denoised = self.transform_spatial_mask_to_node_mask(mask1, enhanced_denoised_features)
            node_mask_original = self.transform_spatial_mask_to_node_mask(mask2, enhanced_original_features)
            
            enhanced_diff_features = (node_mask_denoised * enhanced_denoised_features) - (node_mask_original * enhanced_original_features)
            
            grad_enhancement = self.node_features_to_latents(enhanced_diff_features, batch_info)
            
            hypergraph_loss = F.mse_loss(enhanced_denoised_features, enhanced_original_features)
            
            grad_enhancement = grad_enhancement.to(original_dtype)
        
        if iteration % 50 == 0:
            with torch.no_grad():
                original_diff = denoised_latents_float - original_latents_float
                self.log_enhancement_statistics(grad_enhancement, original_hypergraph_data, denoised_hypergraph_data, iteration, original_diff)
        
        return grad_enhancement, hypergraph_loss
    
    def log_enhancement_statistics(self, grad_enhancement, original_hypergraph_data, denoised_hypergraph_data, iteration, original_diff):
        with torch.no_grad():
            enhanced_diff_norm = torch.norm(grad_enhancement).item()
            enhanced_diff_mean = grad_enhancement.abs().mean().item()
            enhanced_diff_std = grad_enhancement.std().item()
            
            original_diff_norm = torch.norm(original_diff).item()
            original_diff_mean = original_diff.abs().mean().item()
            original_diff_std = original_diff.std().item()
            
            enhancement_ratio = enhanced_diff_norm / (original_diff_norm + 1e-8)
            
            print(f"Iteration {iteration}:")
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

class SimplifiedDHGLatentHypergraph(nn.Module):
    def __init__(self, device='cuda', reconstruction_interval=10, top_k=16, min_hyperedge_size=2):
        super().__init__()
        self.device = device
        self.reconstruction_interval = reconstruction_interval
        self.top_k = top_k
        self.min_hyperedge_size = min_hyperedge_size
        self.last_reconstruction_iter = -1
        self.cached_hypergraph = None
        
        self.encoder_transform = nn.PixelUnshuffle(downscale_factor=2)
        self.decoder_transform = nn.PixelShuffle(upscale_factor=2)
        
        self.hgnn_conv = HGNNPConv(16, 16).to(device)
        
        for param in self.parameters():
            param.requires_grad = False
            
    def encode_latents(self, latents):
        return self.encoder_transform(latents)
    
    def decode_features(self, features):
        return self.decoder_transform(features)
    
    def should_reconstruct_hypergraph(self, iteration):
        return (iteration - self.last_reconstruction_iter) >= self.reconstruction_interval
    
    def create_similarity_threshold_hypergraph_v2(self, node_features, iteration):
        if not self.should_reconstruct_hypergraph(iteration) and self.cached_hypergraph is not None:
            return self.cached_hypergraph
        
        num_nodes, feature_dim = node_features.shape
        similarity_threshold = 0.7
        
        hyperedge_list = []
        hyperedge_weights = []
        
        with torch.no_grad():
            normalized_features = F.normalize(node_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            for i in range(num_nodes):
                similar_mask = similarity_matrix[i] > similarity_threshold
                similar_indices = torch.where(similar_mask)[0]
                
                if len(similar_indices) >= 2 and len(similar_indices) <= 20:
                    similar_nodes = similar_indices.tolist()
                    hyperedge_list.append(similar_nodes)
                    
                    similarities = similarity_matrix[i][similar_indices]
                    avg_similarity = similarities.mean().item()
                    hyperedge_weights.append(avg_similarity)

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
        
        if len(hyperedge_list) == 0:
            print("[WARNING] No hyperedges created! Using identity transformation.")
            hyperedge_list = [list(range(num_nodes))]
            hyperedge_weights = [1.0]
        
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
        if not self.should_reconstruct_hypergraph(iteration) and self.cached_hypergraph is not None:
            return self.cached_hypergraph
        
        num_nodes, feature_dim = node_features.shape
        
        hyperedge_list = []
        hyperedge_weights = []
        
        with torch.no_grad():
            normalized_features = F.normalize(node_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            for i in range(num_nodes):
                similarities = similarity_matrix[i]
                
                _, top_k_indices = torch.topk(similarities, min(self.top_k, num_nodes), largest=True)
                
                if len(top_k_indices) >= self.min_hyperedge_size:
                    similar_nodes = top_k_indices.tolist()
                    hyperedge_list.append(similar_nodes)
                    
                    top_k_similarities = similarities[top_k_indices]
                    avg_similarity = top_k_similarities.mean().item()
                    hyperedge_weights.append(avg_similarity)
        
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
        
        if len(hyperedge_list) == 0:
            print("[WARNING] No hyperedges created! Using identity transformation.")
            hyperedge_list = [list(range(num_nodes))]
            hyperedge_weights = [1.0]
        
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
        hypergraph = hypergraph_data['hypergraph']
        
        with torch.no_grad():
            original_dtype = node_features.dtype
            node_features_float = node_features.float()
            
            enhanced_features = self.hgnn_conv(node_features_float, hypergraph)
            
            enhanced_features = enhanced_features.to(original_dtype)
        
        return enhanced_features
    
    def forward(self, original_latents, denoised_latents, iteration):
        original_dtype = original_latents.dtype
        
        with torch.no_grad():
            original_latents_float = original_latents.float()
            denoised_latents_float = denoised_latents.float()
            
            original_encoded = self.encode_latents(original_latents_float)
            denoised_encoded = self.encode_latents(denoised_latents_float)
        
        B, C, H, W = original_encoded.shape
        num_spatial_nodes = B * H * W
        node_features_original = original_encoded.view(B, C, H * W).permute(0, 2, 1).contiguous().view(num_spatial_nodes, C)
        node_features_denoised = denoised_encoded.view(B, C, H * W).permute(0, 2, 1).contiguous().view(num_spatial_nodes, C)
        
        hypergraph_data = self.create_similarity_threshold_hypergraph_v2(node_features_original, iteration)
        
        original_enhanced = self.apply_hypergraph_convolution(hypergraph_data, node_features_original)
        denoised_enhanced = self.apply_hypergraph_convolution(hypergraph_data, node_features_denoised)
        
        with torch.no_grad():
            feature_diff = original_enhanced - denoised_enhanced
            hypergraph_loss = F.mse_loss(original_enhanced, denoised_enhanced)
        
        feature_diff_reshaped = feature_diff.view(B, H, W, C).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            grad_enhancement = self.decode_features(feature_diff_reshaped)
            grad_enhancement = grad_enhancement.to(original_dtype)
        
        return grad_enhancement, hypergraph_loss

class ImprovedStaticDHGLatentHypergraph(nn.Module):
    def __init__(self, device='cuda', reconstruction_interval=10):
        super().__init__()
        self.device = device
        self.reconstruction_interval = reconstruction_interval
        self.last_reconstruction_iter = -1
        self.cached_hypergraph = None
        
        self.encoder_transform = nn.PixelUnshuffle(downscale_factor=2)
        self.decoder_transform = nn.PixelShuffle(upscale_factor=2)
        
        self.spatial_radius = 3
        self.similarity_threshold = 0.8
        self.cross_view_weight = 0.8
        
        self.hgnn_conv = HGNNPConv(16, 16).to(device)
        
        for param in self.parameters():
            param.requires_grad = False
            
    def encode_latents(self, latents):
        return self.encoder_transform(latents)
    
    def decode_features(self, features):
        return self.decoder_transform(features)
    
    def should_reconstruct_hypergraph(self, iteration):
        return (iteration - self.last_reconstruction_iter) >= self.reconstruction_interval
    
    def create_hypergraph_structure(self, encoded_features, iteration):
        if not self.should_reconstruct_hypergraph(iteration) and self.cached_hypergraph is not None:
            return self.cached_hypergraph
        
        B, C, H, W = encoded_features.shape
        num_nodes = B * H * W
        
        node_features = encoded_features.view(B, C, H*W).permute(0, 2, 1).contiguous().view(num_nodes, C)
        
        hyperedge_list = []
        hyperedge_weights = []
        
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
        
        if B > 1:
            step = max(1, H // 8)
            for i in range(0, H, step):
                for j in range(0, W, step):
                    cross_view_edge = [b * H * W + i * W + j for b in range(B)]
                    hyperedge_list.append(cross_view_edge)
                    hyperedge_weights.append(self.cross_view_weight)
        
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
        hypergraph = hypergraph_data['hypergraph']
        B, C, H, W = hypergraph_data['shape_info']
        
        node_features = input_features.view(B, C, H*W).permute(0, 2, 1).contiguous().view(B*H*W, C)
        
        with torch.no_grad():
            original_dtype = node_features.dtype
            node_features_float = node_features.float()
            
            enhanced_features = self.hgnn_conv(node_features_float, hypergraph)
            
            enhanced_features = enhanced_features.to(original_dtype)
        
        enhanced_features = enhanced_features.view(B, H, W, C).permute(0, 3, 1, 2)
        return enhanced_features
    
    def forward(self, original_latents, denoised_latents, iteration):
        original_dtype = original_latents.dtype
        
        with torch.no_grad():
            original_latents_float = original_latents.float()
            denoised_latents_float = denoised_latents.float()
            
            original_encoded = self.encode_latents(original_latents_float)
            denoised_encoded = self.encode_latents(denoised_latents_float)
        
        hypergraph_data = self.create_hypergraph_structure(original_encoded, iteration)
        
        original_enhanced = self.apply_static_hypergraph_processing(hypergraph_data, original_encoded)
        denoised_enhanced = self.apply_static_hypergraph_processing(hypergraph_data, denoised_encoded)
        
        with torch.no_grad():
            feature_diff = original_enhanced - denoised_enhanced
            
            hypergraph_loss = F.mse_loss(original_enhanced, denoised_enhanced)
        
        with torch.no_grad():
            grad_enhancement = self.decode_features(feature_diff)
            
            grad_enhancement = grad_enhancement.to(original_dtype)
        
        return grad_enhancement, hypergraph_loss

class StaticGradientHypergraphEnhancer:
    def __init__(self, patch_size=4, alpha=0.5, similarity_threshold=0.8, device='cuda'):
        self.patch_size = patch_size
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        print(f"[INFO] Initialized StaticGradientHypergraphEnhancer with patch_size={patch_size}, alpha={alpha}")

    def build_and_average(self, patches: torch.Tensor, num_patches_h, num_patches_w):
        N, P_dim = patches.shape
        hyperedges = []

        norm_patches = F.normalize(patches, p=2, dim=1)
        sim_matrix = torch.matmul(norm_patches, norm_patches.t())

        for i in range(N):
            similar_nodes = torch.where(sim_matrix[i] > self.similarity_threshold)[0]
            if len(similar_nodes) > 1:
                hyperedges.append(similar_nodes.tolist())

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < num_patches_h and 0 <= nj < num_patches_w:
                            neighbors.append(ni * num_patches_w + nj)
                if len(neighbors) > 1:
                    hyperedges.append(neighbors)
        
        if not hyperedges:
            return patches

        enhanced_patches = torch.zeros_like(patches)
        counts = torch.zeros(N, 1, device=self.device)

        unique_hyperedges = list(set(map(tuple, hyperedges)))

        for edge_nodes in unique_hyperedges:
            edge_nodes_tensor = torch.tensor(edge_nodes, device=self.device, dtype=torch.long)
            
            mean_patch = torch.mean(patches[edge_nodes_tensor], dim=0)
            
            enhanced_patches[edge_nodes_tensor] += mean_patch
            counts[edge_nodes_tensor] += 1
        
        counts[counts == 0] = 1
        
        averaged_patches = enhanced_patches / counts
        
        final_patches = torch.where(counts > 1, averaged_patches, patches)
        
        return final_patches

    @torch.no_grad()
    def __call__(self, grad_latent: torch.Tensor):
        B, C, H, W = grad_latent.shape
        p = self.patch_size
        num_patches_h, num_patches_w = H // p, W // p

        patches = rearrange(grad_latent, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        
        enhanced_patches_batch = []
        for i in range(B):
            single_patches = patches[i]
            enhanced_single_patches = self.build_and_average(single_patches, num_patches_h, num_patches_w)
            enhanced_patches_batch.append(enhanced_single_patches)
        
        enhanced_patches = torch.stack(enhanced_patches_batch, dim=0)

        reconstructed_grad = rearrange(
            enhanced_patches, 
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=num_patches_h, w=num_patches_w, p1=p, p2=p, c=C
        )

        final_grad = self.alpha * reconstructed_grad + (1 - self.alpha) * grad_latent
        
        orig_norm = torch.norm(grad_latent, p=2, dim=(1,2,3), keepdim=True)
        final_norm = torch.norm(final_grad, p=2, dim=(1,2,3), keepdim=True)
        
        scale = torch.clamp(orig_norm / (final_norm + 1e-9), min=0.5, max=1.5)
        final_grad = final_grad * scale
        
        return final_grad
