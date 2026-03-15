import torch
import torch.nn.functional as F
import dhg
from dhg.nn import HGNNPConv
from einops import rearrange
import torch.nn as nn
import numpy as np

class DirectSimilarityDHGLatentHypergraph(nn.Module):
    """
    Direct similarity computation version - removed encoding/decoding and block computation
    First apply hypergraph convolution to denoised_latents and original_latents separately, then compute the difference
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
        
        # Directly process 4-channel features
        self.hgnn_conv = HGNNPConv(4, 4).to(device)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def should_reconstruct_hypergraph(self, iteration):
        """Determine whether the hypergraph needs to be reconstructed"""
        return (iteration - self.last_reconstruction_iter) >= self.reconstruction_interval
    
    def latents_to_node_features(self, latents):
        """
        Convert latents to a node feature matrix
        Args:
            latents: [B, 4, 64, 64] input latents
        Returns:
            node_features: [B*64*64, 4] node feature matrix
            batch_info: information for reconstruction
        """
        B, C, H, W = latents.shape
        
        # Reshape to node features: [B, 4, 64, 64] -> [B*64*64, 4]
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
        Convert node features back to latents format
        Args:
            node_features: [B*64*64, 4] node feature matrix
            batch_info: reconstruction information
        Returns:
            latents: [B, 4, 64, 64] reconstructed latents
        """
        B = batch_info['batch_size']
        C = batch_info['channels']
        H = batch_info['height']
        W = batch_info['width']
        
        # Reshape back to original format: [B*64*64, 4] -> [B, 4, 64, 64]
        latents = node_features.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return latents
    
    def create_topk_hypergraph(self, node_features, iteration, batch_info, hypergraph_type="original"):
        """
        Build a hypergraph using the Top-K method. For each node, select the K-1 most similar nodes to form a hyperedge together.

        Args:
            node_features (torch.Tensor): [N, C] node feature matrix, N = B*H*W.
            iteration (int): current iteration count.
            batch_info (dict): dict containing batch_size, height, width, total_nodes.
            hypergraph_type (str): "original" or "denoised", used for differentiation and caching.
        """
        # 1. --- Cache logic (same as before) ---
        # Select cache based on type
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
            # 2. --- Similarity matrix computation (same as before) ---
            # print(f"[DEBUG] Computing full similarity matrix for {hypergraph_type}...")
            normalized_features = F.normalize(node_features, p=2, dim=1)
            
            # Direct matrix multiplication - still a major memory consumer
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            # print(f"[DEBUG] {hypergraph_type} similarity matrix computed, shape: {similarity_matrix.shape}")
            
            # =====================================================================
            # 3. --- [Core change] Build hyperedges based on Top-K ---
            # =====================================================================
            # print(f"[DEBUG] Building hyperedges for {hypergraph_type} using Top-K (k={self.top_k})...")
            
            # Diagonal elements (self-similarity) may be 1, which can affect top-k results; can be set to -inf
            # However, including self as part of a hyperedge is usually reasonable, so we keep it
            
            # Compute Top-K neighbors for all nodes at once - more efficient
            # torch.topk returns two tensors: (top_values, top_indices)
            top_k_similarities, top_k_indices = torch.topk(
                similarity_matrix, 
                k=self.top_k, 
                dim=1, 
                largest=True
            )
            
            # print(f"[DEBUG] Top-K indices calculated, shape: {top_k_indices.shape}")

            # Convert results to hyperedge list and weight list
            # Each row top_k_indices[i] is a hyperedge centered on node i
            hyperedge_list = top_k_indices.tolist()
            
            # Use the mean of these top-k similarities as the weight for each hyperedge
            hyperedge_weights = top_k_similarities.mean(dim=1).tolist()
            
            # print(f"[DEBUG] {hypergraph_type} initial hyperedges created: {len(hyperedge_list)}")
            # =====================================================================

            # 4. --- Remove duplicate hyperedges (unchanged, still good practice) ---
            # print(f"[DEBUG] Removing duplicate hyperedges for {hypergraph_type}...")
            unique_hyperedges = []
            unique_weights = []
            seen_edges = set()
            
            for edge, weight in zip(hyperedge_list, hyperedge_weights):
                # Sort the hyperedge to create a unique identifier
                sorted_edge = tuple(sorted(edge))
                if sorted_edge not in seen_edges:
                    seen_edges.add(sorted_edge)
                    unique_hyperedges.append(edge)
                    unique_weights.append(weight)
            
            hyperedge_list = unique_hyperedges
            hyperedge_weights = unique_weights
            
            # print(f"[DEBUG] {hypergraph_type} final unique hyperedges: {len(hyperedge_list)}")
            
            if len(hyperedge_list) > 0:
                # For Top-K, the average size should theoretically be close to K
                avg_edge_size = np.mean([len(edge) for edge in hyperedge_list])
                # print(f"[DEBUG] {hypergraph_type} average hyperedge size: {avg_edge_size:.2f}")

            # 5. --- Fallback strategy (no longer necessary for Top-K) ---
            # The Top-K method always generates a hyperedge for each node, so hyperedge_list being empty is nearly impossible.
            # Therefore, the previous complex fallback logic can be safely removed, making the code cleaner.
            if len(hyperedge_list) == 0:
                print(f"[CRITICAL WARNING] Top-K method failed to produce any hyperedges. This is highly unexpected.")
                # Build a minimal connected graph as a last resort
                hyperedge_list.append(list(range(self.top_k)))
                hyperedge_weights.append(0.5)

            # 6. --- Create and cache DHG hypergraph (same as before) ---
            # print(f"[DEBUG] Creating DHG hypergraph for {hypergraph_type}...")
            hypergraph = dhg.Hypergraph(num_nodes, hyperedge_list, device=self.device)
            
            hypergraph_data = {
                'hypergraph': hypergraph,
                'edge_weights': torch.tensor(hyperedge_weights, device=self.device),
                'num_nodes': num_nodes,
                'num_edges': len(hyperedge_list),
                'avg_edge_size': np.mean([len(edge) for edge in hyperedge_list]) if hyperedge_list else 0,
                'batch_info': batch_info,
                'k_used': self.top_k,  # record the K value used
                'type': hypergraph_type
            }
            
            # Cache based on type
            if hypergraph_type == "original":
                self.cached_original_hypergraph = hypergraph_data
            else:
                self.cached_denoised_hypergraph = hypergraph_data
            
            self.last_reconstruction_iter = iteration
            
            # print(f"[DEBUG] {hypergraph_type} Top-K hypergraph created successfully!")
            return hypergraph_data
    
    def create_direct_similarity_hypergraph(self, node_features, iteration, batch_info, hypergraph_type="original"):
        """
        Directly compute the similarity matrix - no block decomposition
        Args:
            node_features: [16384, 4] node feature matrix
            iteration: current iteration count
            batch_info: batch information
            hypergraph_type: "original" or "denoised"
        """
        # Select cache based on type
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
            # Directly compute the full similarity matrix - computed all at once, no blocks
            # print(f"[DEBUG] Computing full similarity matrix for {hypergraph_type}...")
            normalized_features = F.normalize(node_features, p=2, dim=1)
            
            # Direct matrix multiplication - may consume a lot of memory
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            # print(f"[DEBUG] {hypergraph_type} similarity matrix computed, shape: {similarity_matrix.shape}")
            # print(f"[DEBUG] Memory usage: {similarity_matrix.numel() * 4 / 1024 / 1024:.2f} MB")
            
            # Build hyperedges based on similarity threshold
            # print(f"[DEBUG] Building hyperedges for {hypergraph_type} with threshold {self.similarity_threshold}...")
            edges_created = 0
            
            for i in range(num_nodes):
                # Find all nodes with similarity to node i above the threshold
                similar_mask = similarity_matrix[i] > self.similarity_threshold
                similar_indices = torch.where(similar_mask)[0]
                
                # Filter by hyperedge size
                if len(similar_indices) >= self.min_hyperedge_size and len(similar_indices) <= self.max_hyperedge_size:
                    similar_nodes = similar_indices.tolist()
                    hyperedge_list.append(similar_nodes)
                    
                    # Use similarity scores as weights
                    similarities = similarity_matrix[i][similar_indices]
                    avg_similarity = similarities.mean().item()
                    hyperedge_weights.append(avg_similarity)
                    edges_created += 1
                
                # Print progress
                if i % 2000 == 0:
                    progress = i / num_nodes * 100
                    # print(f"[DEBUG] {hypergraph_type} Progress: {progress:.1f}%, edges created: {edges_created}")
            
            # print(f"[DEBUG] {hypergraph_type} initial hyperedges created: {len(hyperedge_list)}")
            
            # Remove duplicate hyperedges
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
            
            # Fallback strategy
            if len(hyperedge_list) == 0:
                print(f"[WARNING] No hyperedges created for {hypergraph_type}! Using fallback strategy.")
                # Retry with lower threshold
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
            
            # Minimal fallback
            if len(hyperedge_list) == 0:
                print(f"[WARNING] Still no hyperedges for {hypergraph_type}! Creating minimal fallback.")
                for i in range(min(10, num_nodes // 1000)):
                    start_idx = i * 1000
                    end_idx = min(start_idx + 100, num_nodes)
                    if end_idx - start_idx >= self.min_hyperedge_size:
                        fallback_edge = list(range(start_idx, end_idx))
                        hyperedge_list.append(fallback_edge)
                        hyperedge_weights.append(0.5)
            
            # Create DHG hypergraph
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
            
            # Cache based on type
            if hypergraph_type == "original":
                self.cached_original_hypergraph = hypergraph_data
            else:
                self.cached_denoised_hypergraph = hypergraph_data
            
            self.last_reconstruction_iter = iteration
            
            # print(f"[DEBUG] {hypergraph_type} hypergraph created successfully!")
            return hypergraph_data
    
    def apply_hypergraph_convolution(self, hypergraph_data, node_features):
        """Apply hypergraph convolution"""
        hypergraph = hypergraph_data['hypergraph']
        
        with torch.no_grad():
            original_dtype = node_features.dtype
            node_features_float = node_features.float()
            
            enhanced_features = self.hgnn_conv(node_features_float, hypergraph)
            enhanced_features = enhanced_features.to(original_dtype)
        
        return enhanced_features
    
    def transform_spatial_mask_to_node_mask(self, spatial_mask, node_features):
        """
        Transform a spatial mask [B, 1, H, W] into a node mask [N, D].
        """
        num_nodes, feature_dim = node_features.shape
        
        # Flatten the spatial mask to match node order
        flattened_mask = spatial_mask.view(num_nodes)
        
        # Expand to match feature dimension
        node_mask = flattened_mask.unsqueeze(1).repeat(1, feature_dim)
        
        return node_mask
    
    def forward(self, original_latents, denoised_latents, iteration, target_grad_norm=None, mask1=None, mask2=None):
        """
        Forward pass - first apply hypergraph convolution to both latents separately, then compute the difference
        Args:
            original_latents: [B, 4, 64, 64] original latents
            denoised_latents: [B, 4, 64, 64] denoised latents
            iteration: current iteration count
            target_grad_norm: target gradient magnitude
        """
        original_dtype = original_latents.dtype
        
        with torch.no_grad():
            # 1. Convert to node features (no encoding/decoding)
            original_latents_float = original_latents.float()
            denoised_latents_float = denoised_latents.float()
            
            original_node_features, batch_info = self.latents_to_node_features(original_latents_float)
            denoised_node_features, _ = self.latents_to_node_features(denoised_latents_float)
            
            # print(f"[DEBUG] Node features shape: {original_node_features.shape}")
            
            # 2. Build two hypergraphs separately
            original_hypergraph_data = self.create_topk_hypergraph(
                original_node_features, iteration, batch_info, hypergraph_type="original"
            )
            denoised_hypergraph_data = self.create_topk_hypergraph(
                denoised_node_features, iteration, batch_info, hypergraph_type="denoised"
            )
            
            # 3. Apply hypergraph convolution to each latent separately
            enhanced_original_features = self.apply_hypergraph_convolution(original_hypergraph_data, original_node_features)
            enhanced_denoised_features = self.apply_hypergraph_convolution(denoised_hypergraph_data, denoised_node_features)
            
            # 4. Compute the enhanced difference
            # print(f"mask1 shape: {mask1.shape}")
            # print(f"mask2 shape: {mask2.shape}")
            # print(f"enhanced_denoised_features shape: {enhanced_denoised_features.shape}")
            # print(f"enhanced_original_features shape: {enhanced_original_features.shape}")

            node_mask_denoised = self.transform_spatial_mask_to_node_mask(mask1, enhanced_denoised_features)
            node_mask_original = self.transform_spatial_mask_to_node_mask(mask2, enhanced_original_features)
            
            enhanced_diff_features = (node_mask_denoised * enhanced_denoised_features) - (node_mask_original * enhanced_original_features)
            # enhanced_diff_features = enhanced_denoised_features - enhanced_original_features
            
            # 5. Convert back to latents format
            grad_enhancement = self.node_features_to_latents(enhanced_diff_features, batch_info)
            
            # 6. Compute loss - based on enhanced features vs original features
            # original_loss = F.mse_loss(enhanced_original_features, original_node_features)
            # denoised_loss = F.mse_loss(enhanced_denoised_features, denoised_node_features)
            # hypergraph_loss = (original_loss + denoised_loss) / 2
            
            hypergraph_loss = F.mse_loss(enhanced_denoised_features, enhanced_original_features)
            
            grad_enhancement = grad_enhancement.to(original_dtype)
        
        # 7. Record statistics
        if iteration % 50 == 0:
            with torch.no_grad():
                # Compute original difference for comparison
                original_diff = denoised_latents_float - original_latents_float
                self.log_enhancement_statistics(grad_enhancement, original_hypergraph_data, denoised_hypergraph_data, iteration, original_diff)
        
        return grad_enhancement, hypergraph_loss
    
    def log_enhancement_statistics(self, grad_enhancement, original_hypergraph_data, denoised_hypergraph_data, iteration, original_diff):
        """Record enhancement statistics"""
        with torch.no_grad():
            # Statistics of the enhanced difference
            enhanced_diff_norm = torch.norm(grad_enhancement).item()
            enhanced_diff_mean = grad_enhancement.abs().mean().item()
            enhanced_diff_std = grad_enhancement.std().item()
            
            # Statistics of the original difference
            original_diff_norm = torch.norm(original_diff).item()
            original_diff_mean = original_diff.abs().mean().item()
            original_diff_std = original_diff.std().item()
            
            # Compute enhancement effect
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
    Simplified DHG hypergraph processor - build hyperedges using top-k similarity
    """
    
    def __init__(self, device='cuda', reconstruction_interval=10, top_k=16, min_hyperedge_size=2):
        super().__init__()
        self.device = device
        self.reconstruction_interval = reconstruction_interval
        self.top_k = top_k  # select top-k most similar nodes for each node
        self.min_hyperedge_size = min_hyperedge_size  # minimum hyperedge size
        self.last_reconstruction_iter = -1
        self.cached_hypergraph = None
        
        # Fixed encoder and decoder (no learnable parameters)
        self.encoder_transform = nn.PixelUnshuffle(downscale_factor=2)
        self.decoder_transform = nn.PixelShuffle(upscale_factor=2)
        
        # Fixed hypergraph convolution layers (parameters not updated)
        self.hgnn_conv = HGNNPConv(16, 16).to(device)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def encode_latents(self, latents):
        """Encode: [B, 4, 64, 64] -> [B, 16, 32, 32]"""
        return self.encoder_transform(latents)
    
    def decode_features(self, features):
        """Decode: [B, 16, 32, 32] -> [B, 4, 64, 64]"""
        return self.decoder_transform(features)
    
    def should_reconstruct_hypergraph(self, iteration):
        """Determine whether the hypergraph needs to be reconstructed"""
        return (iteration - self.last_reconstruction_iter) >= self.reconstruction_interval
    
    def create_similarity_threshold_hypergraph_v2(self, node_features, iteration):
        """
        Modified version: use similarity threshold instead of top-k
        """
        if not self.should_reconstruct_hypergraph(iteration) and self.cached_hypergraph is not None:
            return self.cached_hypergraph
        
        # print(f"[DEBUG] Reconstructing similarity threshold hypergraph at iteration {iteration}")
        
        num_nodes, feature_dim = node_features.shape
        similarity_threshold = 0.7  # tunable parameter
        
        hyperedge_list = []
        hyperedge_weights = []
        
        with torch.no_grad():
            # Compute similarity matrix of node features
            normalized_features = F.normalize(node_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            # Use similarity threshold instead of top-k
            for i in range(num_nodes):
                # Find nodes with similarity above the threshold
                similar_mask = similarity_matrix[i] > similarity_threshold
                similar_indices = torch.where(similar_mask)[0]
                
                # Limit hyperedge size
                if len(similar_indices) >= 2 and len(similar_indices) <= 20:
                    similar_nodes = similar_indices.tolist()
                    hyperedge_list.append(similar_nodes)
                    
                    # Use similarity scores as weights
                    similarities = similarity_matrix[i][similar_indices]
                    avg_similarity = similarities.mean().item()
                    hyperedge_weights.append(avg_similarity)

        # Remove duplicate hyperedges (optional)
        unique_hyperedges = []
        unique_weights = []
        seen_edges = set()
        
        for edge, weight in zip(hyperedge_list, hyperedge_weights):
            # Sort the hyperedge for comparison
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
            # If no hyperedges, create one containing all nodes
            hyperedge_list = [list(range(num_nodes))]
            hyperedge_weights = [1.0]
        
        # Create DHG hypergraph
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
        """Build hypergraph structure based on top-k similarity"""
        if not self.should_reconstruct_hypergraph(iteration) and self.cached_hypergraph is not None:
            return self.cached_hypergraph
        
        # print(f"[DEBUG] Reconstructing top-k similarity hypergraph at iteration {iteration}")
        
        num_nodes, feature_dim = node_features.shape
        # print(f"[DEBUG] Node features shape: {node_features.shape}, using top-k={self.top_k}")
        
        hyperedge_list = []
        hyperedge_weights = []
        
        # Build hyperedges based on top-k feature similarity
        with torch.no_grad():
            # Compute similarity matrix of node features
            normalized_features = F.normalize(node_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            # Method 1: find top-k similar nodes for each node and build a hyperedge
            for i in range(num_nodes):
                # Get similarity of node i with all other nodes
                similarities = similarity_matrix[i]
                
                # Find the top-k most similar nodes (including self)
                _, top_k_indices = torch.topk(similarities, min(self.top_k, num_nodes), largest=True)
                
                if len(top_k_indices) >= self.min_hyperedge_size:
                    similar_nodes = top_k_indices.tolist()
                    hyperedge_list.append(similar_nodes)
                    
                    # Use top-k similarity scores as weights
                    top_k_similarities = similarities[top_k_indices]
                    avg_similarity = top_k_similarities.mean().item()
                    hyperedge_weights.append(avg_similarity)
        
        # Remove duplicate hyperedges (optional)
        unique_hyperedges = []
        unique_weights = []
        seen_edges = set()
        
        for edge, weight in zip(hyperedge_list, hyperedge_weights):
            # Sort the hyperedge for comparison
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
            # If no hyperedges, create one containing all nodes
            hyperedge_list = [list(range(num_nodes))]
            hyperedge_weights = [1.0]
        
        # Create DHG hypergraph
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
        """Apply hypergraph convolution"""
        hypergraph = hypergraph_data['hypergraph']
        
        with torch.no_grad():
            # Ensure consistent data type
            original_dtype = node_features.dtype
            node_features_float = node_features.float()
            
            # Apply hypergraph convolution
            enhanced_features = self.hgnn_conv(node_features_float, hypergraph)
            
            # Convert back to original data type
            enhanced_features = enhanced_features.to(original_dtype)
        
        return enhanced_features
    
    def forward(self, original_latents, denoised_latents, iteration):
        """
        Forward pass
        Args:
            original_latents: original latents [B, 4, 64, 64]
            denoised_latents: denoised latents [B, 4, 64, 64]
            iteration: current iteration count
        """
        # Record original data type
        original_dtype = original_latents.dtype
        
        # 1. Encode latents to higher dimension
        with torch.no_grad():
            original_latents_float = original_latents.float()
            denoised_latents_float = denoised_latents.float()
            
            original_encoded = self.encode_latents(original_latents_float)  # [B, 16, 32, 32]
            denoised_encoded = self.encode_latents(denoised_latents_float)  # [B, 16, 32, 32]
        
        # 2. Reshape to node feature matrix
        B, C, H, W = original_encoded.shape
        num_spatial_nodes = B * H * W  # number of spatial nodes
        node_features_original = original_encoded.view(B, C, H * W).permute(0, 2, 1).contiguous().view(num_spatial_nodes, C)  # [B*H*W, 16]
        node_features_denoised = denoised_encoded.view(B, C, H * W).permute(0, 2, 1).contiguous().view(num_spatial_nodes, C)   # [B*H*W, 16]
        
        # print(f"[DEBUG] Node features shape: {node_features_original.shape}")
        
        # 3. Build top-k similarity hypergraph
        hypergraph_data = self.create_similarity_threshold_hypergraph_v2(node_features_original, iteration)
        
        # 4. Apply hypergraph convolution
        original_enhanced = self.apply_hypergraph_convolution(hypergraph_data, node_features_original)
        denoised_enhanced = self.apply_hypergraph_convolution(hypergraph_data, node_features_denoised)
        
        # 5. Compute feature difference
        with torch.no_grad():
            feature_diff = original_enhanced - denoised_enhanced
            hypergraph_loss = F.mse_loss(original_enhanced, denoised_enhanced)
        
        # 6. Reshape back to original format and decode
        feature_diff_reshaped = feature_diff.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, 16, 32, 32]
        
        with torch.no_grad():
            grad_enhancement = self.decode_features(feature_diff_reshaped)  # [B, 4, 64, 64]
            grad_enhancement = grad_enhancement.to(original_dtype)
        
        return grad_enhancement, hypergraph_loss

class ImprovedStaticDHGLatentHypergraph(nn.Module):
    """
    Static DHG hypergraph processor - fixes data type mismatch
    """
    
    def __init__(self, device='cuda', reconstruction_interval=10):
        super().__init__()
        self.device = device
        self.reconstruction_interval = reconstruction_interval
        self.last_reconstruction_iter = -1
        self.cached_hypergraph = None
        
        # Fixed encoder and decoder (no learnable parameters)
        self.encoder_transform = nn.PixelUnshuffle(downscale_factor=2)
        self.decoder_transform = nn.PixelShuffle(upscale_factor=2)
        
        # Fixed hypergraph construction parameters
        self.spatial_radius = 3
        self.similarity_threshold = 0.8
        self.cross_view_weight = 0.8
        
        # Fixed hypergraph convolution layers (parameters not updated)
        self.hgnn_conv = HGNNPConv(16, 16).to(device)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def encode_latents(self, latents):
        """Optimized fixed encoder: [B, 4, 64, 64] -> [B, 16, 32, 32]"""
        return self.encoder_transform(latents)
    
    def decode_features(self, features):
        """Optimized fixed decoder: [B, 16, 32, 32] -> [B, 4, 64, 64]"""
        return self.decoder_transform(features)
    
    def should_reconstruct_hypergraph(self, iteration):
        """Determine whether the hypergraph needs to be reconstructed"""
        return (iteration - self.last_reconstruction_iter) >= self.reconstruction_interval
    
    def create_hypergraph_structure(self, encoded_features, iteration):
        """Build DHG hypergraph structure (periodically reconstructed)"""
        if not self.should_reconstruct_hypergraph(iteration) and self.cached_hypergraph is not None:
            return self.cached_hypergraph
        
        # print(f"[DEBUG] Reconstructing hypergraph at iteration {iteration}")
        
        B, C, H, W = encoded_features.shape
        num_nodes = B * H * W
        
        node_features = encoded_features.view(B, C, H*W).permute(0, 2, 1).contiguous().view(num_nodes, C)
        
        hyperedge_list = []
        hyperedge_weights = []
        
        # 1. Spatial neighborhood hyperedges
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
        
        # 2. Feature similarity hyperedges (simplified)
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
        
        # 3. Cross-view connection hyperedges
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
        """Apply static hypergraph processing (no parameter updates) - fix data type issue"""
        hypergraph = hypergraph_data['hypergraph']
        B, C, H, W = hypergraph_data['shape_info']
        
        node_features = input_features.view(B, C, H*W).permute(0, 2, 1).contiguous().view(B*H*W, C)
        
        with torch.no_grad():
            # Ensure consistent data type - convert to float32 for computation
            original_dtype = node_features.dtype
            node_features_float = node_features.float()
            
            enhanced_features = self.hgnn_conv(node_features_float, hypergraph)
            
            # Convert back to original data type
            enhanced_features = enhanced_features.to(original_dtype)
        
        enhanced_features = enhanced_features.view(B, H, W, C).permute(0, 3, 1, 2)
        return enhanced_features
    
    def forward(self, original_latents, denoised_latents, iteration):
        """
        Forward pass - static processing, no parameter learning
        Args:
            original_latents: original latents [B, 4, 64, 64]
            denoised_latents: denoised latents (latents_noisy - pred_noise) [B, 4, 64, 64]
            iteration: current iteration count
        """
        # Record original data type
        original_dtype = original_latents.dtype
        
        # 1. Encode latents (using PixelUnshuffle) - ensure float32 for computation
        with torch.no_grad():
            # Convert to float32 for encoding
            original_latents_float = original_latents.float()
            denoised_latents_float = denoised_latents.float()
            
            original_encoded = self.encode_latents(original_latents_float)
            denoised_encoded = self.encode_latents(denoised_latents_float)
        
        # 2. Build or retrieve cached hypergraph structure
        hypergraph_data = self.create_hypergraph_structure(original_encoded, iteration)
        
        # 3. Apply static hypergraph processing
        original_enhanced = self.apply_static_hypergraph_processing(hypergraph_data, original_encoded)
        denoised_enhanced = self.apply_static_hypergraph_processing(hypergraph_data, denoised_encoded)
        
        # 4. Compute feature difference
        with torch.no_grad():
            # Use the difference between denoised latent and original latent
            feature_diff = original_enhanced - denoised_enhanced
            
            # Compute MSE loss (original latent vs denoised latent)
            hypergraph_loss = F.mse_loss(original_enhanced, denoised_enhanced)
        
        # 5. Decode back to original size (using PixelShuffle)
        with torch.no_grad():
            grad_enhancement = self.decode_features(feature_diff)
            
            # Convert back to original data type
            grad_enhancement = grad_enhancement.to(original_dtype)
        
        return grad_enhancement, hypergraph_loss

class StaticGradientHypergraphEnhancer:
    """
    A static, non-trainable hypergraph gradient enhancer.
    It smooths and structures gradients by averaging gradient patches within the node sets defined by hyperedges.
    """
    def __init__(self, patch_size=4, alpha=0.5, similarity_threshold=0.8, device='cuda'):
        self.patch_size = patch_size
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # As a non-training module, no learnable parameters are needed
        print(f"[INFO] Initialized StaticGradientHypergraphEnhancer with patch_size={patch_size}, alpha={alpha}")

    def build_and_average(self, patches: torch.Tensor, num_patches_h, num_patches_w):
        """
        Build a hypergraph and perform feature averaging directly on hyperedges.
        Args:
            patches (torch.Tensor): [N, P_dim] gradient patches for a single sample (N = H*W, P_dim = C*p*p)
        Returns:
            torch.Tensor: [N, P_dim] enhanced gradient patches
        """
        N, P_dim = patches.shape
        hyperedges = []

        # 1. Build hyperedges based on feature similarity
        # Use L2-normalized features to compute cosine similarity
        norm_patches = F.normalize(patches, p=2, dim=1)
        sim_matrix = torch.matmul(norm_patches, norm_patches.t())

        for i in range(N):
            similar_nodes = torch.where(sim_matrix[i] > self.similarity_threshold)[0]
            if len(similar_nodes) > 1:
                hyperedges.append(similar_nodes.tolist())

        # 2. Build hyperedges based on spatial proximity
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                neighbors = []
                # 3x3 neighborhood
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < num_patches_h and 0 <= nj < num_patches_w:
                            neighbors.append(ni * num_patches_w + nj)
                if len(neighbors) > 1:
                    hyperedges.append(neighbors)
        
        if not hyperedges:  # if no hyperedges, return original patches directly
            return patches

        # 3. Average on hyperedges (simulating hypergraph convolution)
        # Initialize a zero tensor with the same shape as original patches to accumulate results
        enhanced_patches = torch.zeros_like(patches)
        # Create a counter to track how many hyperedges cover each patch
        counts = torch.zeros(N, 1, device=self.device)

        unique_hyperedges = list(set(map(tuple, hyperedges)))  # deduplicate

        for edge_nodes in unique_hyperedges:
            edge_nodes_tensor = torch.tensor(edge_nodes, device=self.device, dtype=torch.long)
            
            # Compute the mean of all patches on this hyperedge
            mean_patch = torch.mean(patches[edge_nodes_tensor], dim=0)
            
            # Add this mean value to all patches belonging to this hyperedge
            enhanced_patches[edge_nodes_tensor] += mean_patch
            counts[edge_nodes_tensor] += 1
        
        # Prevent division by zero; for nodes not covered by any hyperedge, keep original
        counts[counts == 0] = 1
        
        # Compute weighted average
        averaged_patches = enhanced_patches / counts
        
        # For nodes not covered by any hyperedge, use their original values
        uncovered_nodes_mask = (counts.squeeze() == 1)  # counts initialized to 0, incremented only when covered
        # Note: there is a logical subtlety here - a node covered by only one hyperedge will also be averaged.
        # We should find those nodes covered by zero hyperedges.
        # A more robust approach is:
        final_patches = torch.where(counts > 1, averaged_patches, patches)
        
        return final_patches

    @torch.no_grad()  # explicitly indicate this function should not track gradients
    def __call__(self, grad_latent: torch.Tensor):
        B, C, H, W = grad_latent.shape
        p = self.patch_size
        num_patches_h, num_patches_w = H // p, W // p

        # 1. Convert gradient to patch representation
        patches = rearrange(grad_latent, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        
        enhanced_patches_batch = []
        for i in range(B):  # process each sample in the batch separately
            single_patches = patches[i] # [N, P_dim]
            enhanced_single_patches = self.build_and_average(single_patches, num_patches_h, num_patches_w)
            enhanced_patches_batch.append(enhanced_single_patches)
        
        enhanced_patches = torch.stack(enhanced_patches_batch, dim=0) # [B, N, P_dim]

        # 2. Reconstruct gradient
        reconstructed_grad = rearrange(
            enhanced_patches, 
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=num_patches_h, w=num_patches_w, p1=p, p2=p, c=C
        )

        # 3. Residual connection
        final_grad = self.alpha * reconstructed_grad + (1 - self.alpha) * grad_latent
        
        # 4. Stabilization (optional but recommended)
        orig_norm = torch.norm(grad_latent, p=2, dim=(1,2,3), keepdim=True)
        final_norm = torch.norm(final_grad, p=2, dim=(1,2,3), keepdim=True)
        
        scale = torch.clamp(orig_norm / (final_norm + 1e-9), min=0.5, max=1.5)
        final_grad = final_grad * scale
        
        return final_grad