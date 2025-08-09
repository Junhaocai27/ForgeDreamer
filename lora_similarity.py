import torch
import numpy as np
from safetensors import safe_open
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr
import os
from typing import Dict, List, Tuple, Optional
import warnings

# Use 'Agg' backend for non-GUI servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Remove font settings that require special installation
plt.rcParams['axes.unicode_minus'] = False 

class LoRALatentSimilarityAnalyzer:
    def __init__(self):
        self.original_loras = {}
        self.fusion_loras = {}
        self.latent_representations = {}
        
    def load_lora_weights(self, file_path: str, name: str, is_fusion: bool = False) -> Dict[str, torch.Tensor]:
        weights = {}
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
            
            if is_fusion:
                self.fusion_loras[name] = weights
            else:
                self.original_loras[name] = weights
                
            print(f"Successfully loaded {name}: {len(weights)} weight tensors")
            return weights
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def load_multiple_original_loras(self, lora_dir: str, pattern: str = "*.safetensors"):
        lora_path = Path(lora_dir)
        lora_files = list(lora_path.glob(pattern))
        
        print(f"Found {len(lora_files)} LoRA files in {lora_dir}")
        
        for lora_file in lora_files:
            name = lora_file.stem
            self.load_lora_weights(str(lora_file), name, is_fusion=False)
    
    def extract_latent_features(self, weights: Dict[str, torch.Tensor], 
                              method: str = "svd_directional",
                              n_components: int = 16) -> np.ndarray:
        # This print statement is now in compute_all_latent_representations
        if method == "flatten":
            flattened = [weights[key].flatten() for key in sorted(weights.keys())]
            return torch.cat(flattened).numpy()
        
        elif method == "statistical":
            features = []
            for key in sorted(weights.keys()):
                tensor = weights[key].numpy().flatten()
                stats = [
                    np.mean(tensor), np.std(tensor), np.median(tensor),
                    np.min(tensor), np.max(tensor),
                    np.linalg.norm(tensor, ord=1), np.linalg.norm(tensor, ord=2)
                ]
                features.extend(stats)
            return np.array(features, dtype=object) # Return as object to handle varying lengths initially

        elif method == "svd_directional":
            priority_keys = [k for k in weights.keys() if "lora_down" in k and "attn2" in k]
            if not priority_keys:
                priority_keys = [k for k in weights.keys() if "lora_down" in k and "attn" in k]
            if not priority_keys:
                priority_keys = [k for k in weights.keys() if "lora_down" in k]

            if not priority_keys:
                warnings.warn(f"No 'lora_down' weights found. Falling back to 'statistical' method.")
                return self.extract_latent_features(weights, "statistical")
            
            all_matrices = []
            for key in sorted(priority_keys):
                matrix = weights[key].numpy()
                if matrix.ndim == 2:
                    all_matrices.append(matrix.T) 
            
            if not all_matrices:
                 warnings.warn(f"No 2D matrices found among selected weights. Falling back to 'statistical' method.")
                 return self.extract_latent_features(weights, "statistical")

            combined_matrix = np.concatenate(all_matrices, axis=1)
            
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            try:
                latent_features = svd.fit_transform(combined_matrix.T)
                return latent_features.flatten()
            except Exception as e:
                warnings.warn(f"SVD calculation failed: {e}. Falling back to 'statistical' method.")
                return self.extract_latent_features(weights, "statistical")
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")

    def compute_all_latent_representations(self, method: str = "svd_directional", **kwargs):
        print(f"\n--- Computing Latent Representations (Method: {method}) ---")
        self.latent_representations = {}
        
        all_loras = {f"Original_{name}": w for name, w in self.original_loras.items()}
        all_loras.update({f"Fusion_{name}": w for name, w in self.fusion_loras.items()})

        # Step 1: Extract features for all LoRAs
        for name, weights in all_loras.items():
            print(f"  > Extracting features for {name}...")
            features = self.extract_latent_features(weights, method, **kwargs)
            self.latent_representations[name] = features
        
        # Step 2: **FIX** Check for and resolve inconsistent feature lengths
        lengths = {name: len(vec) for name, vec in self.latent_representations.items()}
        if len(set(lengths.values())) > 1:
            min_len = min(lengths.values())
            warnings.warn(f"Inconsistent feature vector lengths detected (e.g., {set(lengths.values())}). "
                          f"This is likely because LoRA files have different numbers of layers. "
                          f"All vectors will be truncated to the shortest length: {min_len}.")
            for name in self.latent_representations:
                self.latent_representations[name] = self.latent_representations[name][:min_len]

        print(f"--- Computation complete. {len(self.latent_representations)} latent representations created. ---\n")
    
    def _get_preprocessed_latent_vector(self, key: str) -> Optional[np.ndarray]:
        if key not in self.latent_representations:
            return None
        
        vec = self.latent_representations[key].copy().astype(np.float64)
        vec -= np.mean(vec)
        norm = np.linalg.norm(vec)
        if norm < 1e-9: return vec
        return vec / norm

    def calculate_fusion_to_originals_similarity(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for fusion_name in self.fusion_loras.keys():
            fusion_key = f"Fusion_{fusion_name}"
            fusion_features = self._get_preprocessed_latent_vector(fusion_key)
            if fusion_features is None: continue
            results[fusion_name] = {}
            for original_name in self.original_loras.keys():
                original_key = f"Original_{original_name}"
                original_features = self._get_preprocessed_latent_vector(original_key)
                if original_features is None: continue
                
                cosine_sim = np.dot(fusion_features, original_features)
                with warnings.catch_warnings(): # Pearson can warn about constant input
                    warnings.simplefilter("ignore")
                    pearson_corr, _ = pearsonr(fusion_features, original_features)
                l2_dist = np.linalg.norm(fusion_features - original_features)
                
                results[fusion_name][original_name] = {
                    'cosine_similarity': float(cosine_sim),
                    'pearson_correlation': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
                    'l2_distance': float(l2_dist)
                }
        return results

    def find_most_similar_originals(self, results: Dict, metric: str = "cosine_similarity", top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        most_similar = {}
        for fusion_name, similarities in results.items():
            reverse = metric != "l2_distance"
            sorted_items = sorted(similarities.items(), key=lambda x: x[1][metric], reverse=reverse)
            most_similar[fusion_name] = [(name, sim[metric]) for name, sim in sorted_items[:top_k]]
        return most_similar

    # FIX: English labels and titles
    def visualize_latent_space(self, save_path: Path, method: str = "pca", n_components: int = 2, title_suffix: str = ""):
        if not self.latent_representations: return
        
        names = list(self.latent_representations.keys())
        features = [self._get_preprocessed_latent_vector(name) for name in names]
        
        valid_features = [f for f in features if f is not None and len(f) > 0]
        valid_names = [name for name, f in zip(names, features) if f is not None and len(f) > 0]

        if len(valid_names) < n_components:
            print(f"Warning: Not enough data points ({len(valid_names)}) for {method.upper()} visualization. Skipping.")
            return

        features_array = np.array(valid_features)
        
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_features = reducer.fit_transform(features_array)
            title = f"LoRA Latent Space via PCA {title_suffix}"
        elif method == "tsne":
            perplexity = min(30, len(valid_names) - 1)
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
            reduced_features = reducer.fit_transform(features_array)
            title = f"LoRA Latent Space via t-SNE {title_suffix}"
        
        plt.figure(figsize=(14, 10))
        original_indices = [i for i, name in enumerate(valid_names) if name.startswith("Original_")]
        fusion_indices = [i for i, name in enumerate(valid_names) if name.startswith("Fusion_")]
        
        plt.scatter(reduced_features[original_indices, 0], reduced_features[original_indices, 1], c='blue', alpha=0.6, s=80, label='Original LoRAs')
        plt.scatter(reduced_features[fusion_indices, 0], reduced_features[fusion_indices, 1], c='red', alpha=0.9, s=150, marker='*', label='Fused LoRAs')
        
        for i, name in enumerate(valid_names):
            display_name = name.replace("Original_", "O_").replace("Fusion_", "F_")
            plt.annotate(display_name, (reduced_features[i, 0], reduced_features[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, weight='bold')
        
        plt.title(title, fontsize=16)
        plt.xlabel("Component 1", fontsize=12)
        plt.ylabel("Component 2", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        print(f"  > Saving latent space plot to: {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # FIX: English labels and titles
    def create_similarity_heatmap(self, results: Dict, save_path: Path, metric: str = "cosine_similarity", title_suffix: str = ""):
        if not results: return
        
        fusion_names = list(results.keys())
        original_names = sorted(list(results[fusion_names[0]].keys()))
        
        matrix = np.array([[results[f][o][metric] for o in original_names] for f in fusion_names])
        
        plt.figure(figsize=(max(10, len(original_names) * 0.8), max(8, len(fusion_names) * 0.8)))
        sns.heatmap(matrix, xticklabels=original_names, yticklabels=fusion_names,
                   annot=True, fmt='.4f', cmap='viridis', cbar_kws={'label': f"{metric.replace('_', ' ').title()}"})
        
        plt.title(f"Similarity of Fused vs. Original LoRAs ({metric.title()}) {title_suffix}", fontsize=16)
        plt.xlabel('Original LoRAs', fontsize=12)
        plt.ylabel('Fused LoRAs', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        print(f"  > Saving heatmap to: {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # FIX: English labels
    def print_detailed_analysis(self, results: Dict, top_k: int = 5):
        print("=" * 80)
        print("LoRA Fusion vs. Original Weights Latent Similarity Analysis")
        print("=" * 80)
        
        for fusion_name, similarities in results.items():
            print(f"\nAnalysis for [{fusion_name}] vs. Original LoRAs (sorted by Cosine Similarity):")
            print("-" * 70)
            
            sorted_by_cosine = sorted(similarities.items(), key=lambda x: x[1]['cosine_similarity'], reverse=True)
            
            for i, (original_name, metrics) in enumerate(sorted_by_cosine, 1):
                cosine_pct = metrics['cosine_similarity'] * 100
                pearson_pct = (metrics['pearson_correlation'] + 1) * 50
                print(f"{i:2d}. {original_name:<30} | Cosine Sim: {cosine_pct:6.2f}% | Pearson Sim: {pearson_pct:6.2f}% | L2 Dist: {metrics['l2_distance']:.4f}")

            print(f"\nTop {top_k} most similar originals for [{fusion_name}] (based on Cosine Similarity):")
            for i, (name, metrics) in enumerate(sorted_by_cosine[:top_k], 1):
                cosine_pct = metrics['cosine_similarity'] * 100
                print(f"  {i}. {name} ({cosine_pct:.2f}%)")
            print()

def main():
    output_dir = Path("./lora_analysis_plots")
    output_dir.mkdir(exist_ok=True)
    print(f"All analysis plots will be saved to: {output_dir.resolve()}")
    
    analyzer = LoRALatentSimilarityAnalyzer()
    
    print("\n--- Step 1: Loading LoRA files ---")
    # --- PLEASE REPLACE WITH YOUR ACTUAL PATHS ---
    original_loras_dir = "/root/lora_weight_before_distill/distill_weight5"
    addition_path = "/root/lora_train/multi_combine_add/resistor_diode2.safetensors"
    distillation_path = "/root/lora_train/multi_combine_20250722_191015/multi_teacher_distilled/final_multi_teacher_hybrid_lora_step_5000.safetensors"
    # --- PLEASE REPLACE WITH YOUR ACTUAL PATHS ---

    analyzer.load_multiple_original_loras(original_loras_dir)
    analyzer.load_lora_weights(addition_path, "LoRA_Addition", is_fusion=True)
    analyzer.load_lora_weights(distillation_path, "LoRA_Distillation", is_fusion=True)

    print("\n\n" + "="*40)
    print("||  Analysis 1: Using 'statistical' method  ||")
    print("="*40)
    analyzer.compute_all_latent_representations(method="statistical")
    results_stat = analyzer.calculate_fusion_to_originals_similarity()
    analyzer.print_detailed_analysis(results_stat)
    analyzer.create_similarity_heatmap(results_stat, save_path=output_dir / "heatmap_statistical.png", metric="cosine_similarity", title_suffix="(Method: statistical)")
    analyzer.visualize_latent_space(save_path=output_dir / "pca_statistical.png", method="pca", title_suffix="(Method: statistical)")

    print("\n\n" + "="*40)
    print("||  Analysis 2: Using 'svd_directional' method (Recommended) ||")
    print("="*40)
    analyzer.compute_all_latent_representations(method="svd_directional", n_components=32) 
    results_svd = analyzer.calculate_fusion_to_originals_similarity()
    
    print("\n--- Detailed Analysis Report (svd_directional) ---")
    analyzer.print_detailed_analysis(results_svd)
    
    print("\n--- Generating Plots (svd_directional) ---")
    analyzer.create_similarity_heatmap(results_svd, save_path=output_dir / "heatmap_svd_directional.png", metric="cosine_similarity", title_suffix="(Method: svd_directional)")
    analyzer.visualize_latent_space(save_path=output_dir / "pca_svd_directional.png", method="pca", title_suffix="(Method: svd_directional)")
    analyzer.visualize_latent_space(save_path=output_dir / "tsne_svd_directional.png", method="tsne", title_suffix="(Method: svd_directional)")
    
    print(f"\nAnalysis complete! All plots have been saved to the '{output_dir.resolve()}' directory.")

if __name__ == "__main__":
    main()