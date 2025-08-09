import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from typing import Dict, List, Tuple
from pathlib import Path
import warnings
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from adjustText import adjust_text

warnings.filterwarnings("ignore")

# --- Global Configuration ---
BASE_MODEL_PATH = "/root/stable-diffusion-2-1-base"
CLIP_MODEL_ID = "/root/LucidDreamer/clip_vit_large_patch14"
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"

class LoRAOutputSimilarityAnalyzer:
    def __init__(self, base_model_path: str, output_dir_path: str = "lora_analysis_output", device: str = DEVICE):
        print(f"--- Initializing Analyzer, will run on {device} ---")
        self.device = device
        
        print(f"Loading base model: {base_model_path}")
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16).to(self.device)
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        except Exception as e:
            raise RuntimeError(f"Failed to load the base model: {base_model_path}\nError: {e}")

        self.feature_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
        self.feature_extractor = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(self.device)
        self.feature_extractor.eval()
        
        self.original_loras: Dict[str, Path] = {}
        self.fusion_loras: Dict[str, Path] = {}
        self.concept_features: Dict[Tuple[str, str], np.ndarray] = {}
        self.lora_triggers: Dict[str, List[str]] = {}
        
        self.output_dir = Path(output_dir_path)
        self.output_dir.mkdir(exist_ok=True)
        print(f"Analysis results will be saved to: {self.output_dir.resolve()}")

        # 设置字体以支持中文
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100

    def add_lora(self, name: str, path: str, trigger_words: List[str] = None, is_fusion: bool = False):
        lora_path = Path(path)
        if not lora_path.exists():
            warnings.warn(f"Warning: LoRA file does not exist, skipped: {path}")
            return
        if trigger_words is None:
            trigger_words = [lora_path.stem.lower()]
        
        self.lora_triggers[name] = trigger_words
        if is_fusion:
            self.fusion_loras[name] = lora_path
            print(f"Registered fusion LoRA: {name}")
        else:
            self.original_loras[name] = lora_path
            print(f"Registered original LoRA: {name}")

    def load_multiple_original_loras(self, lora_dir: str, pattern: str = "*.safetensors"):
        lora_path = Path(lora_dir)
        for f in lora_path.glob(pattern):
            self.add_lora(f.stem, str(f), is_fusion=False)

    def _load_lora_with_patch_pipe(self, lora_path: Path):
        try:
            from lora_diffusion import tune_lora_scale, patch_pipe
            # Always create a fresh pipe instance to avoid state leakage
            pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16).to(self.device)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            patch_pipe(pipe, str(lora_path), patch_text=True, patch_ti=True, patch_unet=True)
            tune_lora_scale(pipe.unet, 1.00); tune_lora_scale(pipe.text_encoder, 1.00)
            self.pipe = pipe
            return True
        except ImportError:
            raise ImportError("The 'lora_diffusion' library is required. Please install it via 'pip install lora-diffusion'.")
        except Exception as e:
            warnings.warn(f"Failed to load LoRA using patch_pipe: {e}")
            return False

    def _extract_trigger_from_prompt(self, prompt: str) -> str:
        match = re.search(r'<([^>]+)>', prompt)
        return match.group(1).lower() if match else ""

    @torch.no_grad()
    def generate_and_extract_features(self, lora_path: Path, prompt: str, seed: int) -> np.ndarray:
        try:
            if not self._load_lora_with_patch_pipe(lora_path):
                return np.array([])
            generator = torch.Generator(self.device).manual_seed(seed)
            image = self.pipe(prompt, num_inference_steps=50, generator=generator, guidance_scale=7.5).images[0]
            inputs = self.feature_processor(images=image, return_tensors="pt").to(self.device)
            return self.feature_extractor(**inputs).image_embeds.cpu().numpy().flatten()
        except Exception as e:
            warnings.warn(f"Error during feature generation: {e}")
            return np.array([])

    def run_analysis(self, prompts: List[str], seeds: List[int]):
        print("\n--- Starting Concept-Based Analysis ---")
        all_lora_paths = {**self.original_loras, **self.fusion_loras}
        concept_all_features = {}

        for prompt in prompts:
            trigger = self._extract_trigger_from_prompt(prompt)
            if not trigger: 
                warnings.warn(f"Could not find a trigger word in prompt: '{prompt}'. Skipping.")
                continue

            for lora_name, lora_path in all_lora_paths.items():
                if trigger in self.lora_triggers.get(lora_name, []):
                    key = (lora_name, trigger)
                    print(f"  > Processing concept: {key}")
                    if key not in concept_all_features:
                        concept_all_features[key] = []
                    
                    for seed in seeds:
                        features = self.generate_and_extract_features(lora_path, prompt, seed)
                        if features.size > 0:
                            concept_all_features[key].append(features)

        self.concept_features = {}
        for key, features_list in concept_all_features.items():
            if features_list:
                self.concept_features[key] = np.mean(features_list, axis=0)
                print(f"✓ Concept '{key}': Averaged {len(features_list)} feature vectors.")
        print("--- Analysis complete ---")

    def calculate_comprehensive_similarity(self):
        """
        计算相似性分析 - 采用简化的“概念对概念匹配”方法。
        """
        if len(self.concept_features) < 2:
            print("Similarity calculation SKIPPED: At least 2 concepts are required.")
            return None, None

        print("\n--- Calculating Concept-to-Concept Similarity Analysis ---")
        
        original_features = {}
        fusion_features = {'Addition': {}, 'Distillation': {}}
        
        for key, features in self.concept_features.items():
            lora_name, trigger = key
            if "Addition" in lora_name: fusion_features['Addition'][trigger] = features
            elif "Distillation" in lora_name: fusion_features['Distillation'][trigger] = features
            else: original_features[trigger] = features
        
        report_lines = [
            "# LoRA Fusion Similarity Analysis Report (Concept-to-Concept)\n",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            "="*80 + "\n", "METHODOLOGY\n", "="*80 + "\n\n",
            "This analysis uses concept-to-concept matching. Each concept from a fused LoRA\n",
            "is directly compared to the corresponding concept from its original LoRA.\n",
            "This provides a direct measure of concept preservation.\n\n",
            "="*80 + "\n", "EXECUTIVE SUMMARY\n", "="*80 + "\n\n"
        ]
        
        concept_to_concept_similarities = {}
        overall_similarities = {}
        
        for fusion_type in ['Addition', 'Distillation']:
            if fusion_type not in fusion_features or not fusion_features[fusion_type]: continue
            
            concept_matching_sims = {}
            for trigger, fusion_feat in fusion_features[fusion_type].items():
                if trigger in original_features:
                    orig_feat = original_features[trigger]
                    sim = cosine_similarity([fusion_feat], [orig_feat])[0][0]
                    concept_matching_sims[trigger] = sim
                else:
                    warnings.warn(f"Warning: No matching original concept found for '{trigger}' in {fusion_type} fusion.")
            
            concept_to_concept_similarities[fusion_type] = concept_matching_sims
            
            if concept_matching_sims:
                values = list(concept_matching_sims.values())
                overall_similarities[fusion_type] = {
                    'mean': np.mean(values), 'std': np.std(values),
                    'min': np.min(values), 'max': np.max(values),
                    'concept_similarities': concept_matching_sims
                }
        
        for fusion_type, stats in overall_similarities.items():
            report_lines.extend([
                f"## {fusion_type} Fusion LoRA Performance:\n",
                f"   - Average Concept Preservation: {stats['mean']:.6f}\n",
                f"   - Consistency (Standard Deviation): {stats['std']:.6f} (lower is better)\n",
                f"   - Performance Range: [{stats['min']:.6f} (worst) - {stats['max']:.6f} (best)]\n",
                f"   - Overall Grade: {self._get_performance_grade(stats['mean'])}\n\n"
            ])
        
        if 'Addition' in overall_similarities and 'Distillation' in overall_similarities:
            add_mean = overall_similarities['Addition']['mean']
            dist_mean = overall_similarities['Distillation']['mean']
            better = 'Addition' if add_mean > dist_mean else 'Distillation'
            report_lines.extend([
                "## Comparative Analysis:\n",
                f"   - Addition Average: {add_mean:.6f}\n",
                f"   - Distillation Average: {dist_mean:.6f}\n",
                f"   - Superior Method: {better} Fusion by {abs(add_mean - dist_mean):.6f}\n\n"
            ])
        
        report_lines.extend(["="*80 + "\n", "DETAILED ANALYSIS\n", "="*80 + "\n\n"])
        
        final_scores = {}
        for fusion_type, stats in overall_similarities.items():
            report_lines.append(f"## {fusion_type} Fusion - Individual Concept Scores:\n")
            sorted_concepts = sorted(stats['concept_similarities'].items(), key=lambda x: x[1], reverse=True)
            for i, (concept, sim) in enumerate(sorted_concepts, 1):
                report_lines.append(f"   {i}. {concept}: {sim:.6f} ({self._get_performance_grade(sim)})\n")
            
            base_score = stats['mean'] * 100
            consistency_penalty = stats['std'] * 50
            final_scores[fusion_type] = max(0, base_score - consistency_penalty)
            report_lines.append("\n" + "-"*60 + "\n\n")

        report_lines.extend(["="*80 + "\n", "FINAL SCORES & RECOMMENDATION\n", "="*80 + "\n\n"])
        
        for fusion_type, stats in overall_similarities.items():
            final_score = final_scores.get(fusion_type, 0)
            report_lines.extend([
                f"## {fusion_type} Fusion Final Assessment:\n",
                f"   - Base Score (from average): {stats['mean'] * 100:.2f}\n",
                f"   - Consistency Penalty (from std dev): -{stats['std'] * 50:.2f}\n",
                f"   - Final Score: {final_score:.2f} / 100\n",
                f"   - Recommendation: {self._get_recommendation(stats['mean'], stats['std'])}\n\n"
            ])
        
        if len(final_scores) >= 2:
            best_method = max(final_scores.items(), key=lambda x: x[1])
            report_lines.extend([
                "## Overall Recommendation:\n",
                f"   Based on the final scores, the best performing method is:\n",
                f"   >> {best_method[0]} Fusion << with a score of {best_method[1]:.2f} / 100.\n",
                f"   Confidence in this recommendation is {self._get_confidence_level(overall_similarities)}.\n\n"
            ])
        
        report_path = self.output_dir / "similarity_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        print(f"✓ Simplified similarity report saved to: {report_path}")
        
        summary_path = self.output_dir / "similarity_scores_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# LoRA Fusion Similarity Summary (Concept-to-Concept)\n\n")
            for fusion_type, stats in overall_similarities.items():
                f.write(f"## {fusion_type} Fusion Results:\n")
                f.write(f"{fusion_type}_average_preservation: {stats['mean']:.8f}\n")
                f.write(f"{fusion_type}_std_deviation: {stats['std']:.8f}\n")
                f.write(f"{fusion_type}_final_score: {final_scores.get(fusion_type, 0):.2f}\n\n")
                f.write(f"### {fusion_type} Individual Scores:\n")
                for concept, sim in stats['concept_similarities'].items():
                    f.write(f"{fusion_type}_{concept}_similarity: {sim:.8f}\n")
                f.write("\n")
            
            if len(final_scores) >= 2:
                best = max(final_scores.items(), key=lambda x: x[1])
                f.write(f"## Conclusion\n")
                f.write(f"Best_Method: {best[0]}_Fusion\n")
                f.write(f"Best_Score: {best[1]:.2f}\n")
        print(f"✓ Scores summary saved to: {summary_path}")
        
        return overall_similarities, concept_to_concept_similarities

    def _get_performance_grade(self, score):
        if score >= 0.9: return "A+ (Excellent)"
        elif score >= 0.8: return "A (Very Good)"
        elif score >= 0.7: return "B (Good)"
        elif score >= 0.6: return "C (Fair)"
        else: return "D (Poor)"
    
    def _get_recommendation(self, mean_sim, std_sim):
        if mean_sim >= 0.8 and std_sim <= 0.1:
            return "Recommended. High fusion quality and consistency."
        elif mean_sim >= 0.7 and std_sim <= 0.15:
            return "Acceptable. Good fusion quality, suitable for general use."
        elif mean_sim >= 0.6:
            return "Use with caution. Moderate quality, may have inconsistencies."
        else:
            return "Not recommended. Poor quality and/or high inconsistency."

    def _get_confidence_level(self, similarities):
        if len(similarities) < 2: return "N/A"
        scores = [stats.get('mean', 0) for stats in similarities.values()]
        diff = abs(scores[0] - scores[1]) if len(scores) >= 2 else 0
        if diff >= 0.05: return "High"
        elif diff >= 0.02: return "Medium"  
        else: return "Low (methods perform very similarly)"

    def create_enhanced_pca_plot(self):
        """
        生成一个不带文字标签的PCA图，以展示概念在特征空间中的分布。
        """
        if len(self.concept_features) < 2:
            print("PCA Plot SKIPPED: At least 2 concept points are required.")
            return

        print("\n--- Generating PCA Visualization (No Labels) ---")
        
        concept_keys = list(self.concept_features.keys())
        features = np.array(list(self.concept_features.values()))
        features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features_normalized)
        
        fig, ax = plt.subplots(figsize=(8, 6))

        style_mapping = {
            'original': {'color': '#2E86AB', 'marker': 'o', 'size': 140, 'label': 'Original LoRA', 'alpha': 0.9},
            'addition': {'color': '#A23B72', 'marker': 's', 'size': 180, 'label': 'Addition Fusion', 'alpha': 0.9},
            'distillation': {'color': '#F18F01', 'marker': '^', 'size': 180, 'label': 'Distillation Fusion', 'alpha': 0.9}
        }

        plotted_labels = set()
        coords_map = {}

        for i, key in enumerate(concept_keys):
            lora_name, _ = key
            coords = features_2d[i]
            coords_map[key] = coords

            style_type = 'original'
            if "Addition" in lora_name: style_type = 'addition'
            elif "Distillation" in lora_name: style_type = 'distillation'

            style = style_mapping[style_type]
            label = style['label'] if style['label'] not in plotted_labels else ""

            ax.scatter(coords[0], coords[1], c=style['color'], marker=style['marker'], s=style['size'],
                    alpha=style['alpha'], edgecolors='black', linewidth=0.5, label=label, zorder=5)

            if label: plotted_labels.add(label)

        original_coords = {key[1]: features_2d[i] for i, key in enumerate(concept_keys) if "Fusion" not in key[0]}
        fusion_coords = {
            'Addition': {key[1]: features_2d[i] for i, key in enumerate(concept_keys) if "Addition" in key[0]},
            'Distillation': {key[1]: features_2d[i] for i, key in enumerate(concept_keys) if "Distillation" in key[0]}
        }

        for trigger, orig_coords in original_coords.items():
            if trigger in fusion_coords['Addition']:
                add_coords = fusion_coords['Addition'][trigger]
                ax.plot([orig_coords[0], add_coords[0]], [orig_coords[1], add_coords[1]],
                        color=style_mapping['addition']['color'], linestyle='--', linewidth=1.2, alpha=0.5, zorder=1)
            if trigger in fusion_coords['Distillation']:
                dist_coords = fusion_coords['Distillation'][trigger]
                ax.plot([orig_coords[0], dist_coords[0]], [orig_coords[1], dist_coords[1]],
                        color=style_mapping['distillation']['color'], linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)

        ax.set_title('LoRA Concept Feature Space Analysis (PCA)', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        
        legend = ax.legend(fontsize=8, markerscale=0.7, loc='best')
        legend.get_frame().set_alpha(0.9)

        plt.tight_layout()
        pca_path = self.output_dir / "pca_plot_no_labels.png"
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ PCA plot without labels saved to: {pca_path}")

    def create_similarity_bar_chart(self, similarity_results: Dict):
        """
        创建一个柱状图，用于直接比较每种融合方法在每个概念上的余弦相似度分数。
        """
        if not similarity_results or not all(k in similarity_results for k in ['Addition', 'Distillation']):
            print("Bar Chart SKIPPED: Incomplete similarity results available.")
            return

        print("\n--- Generating Similarity Score Bar Chart ---")

        add_sims = similarity_results['Addition']['concept_similarities']
        dist_sims = similarity_results['Distillation']['concept_similarities']
        
        # 确保我们有统一的概念列表，并按名称排序
        concept_keys_sorted = sorted(list(add_sims.keys()))
        
        # 使用通用标签 E1, E2, ...
        generic_labels = {key: f"E{i+1}" for i, key in enumerate(concept_keys_sorted)}
        
        addition_scores = [add_sims[key] for key in concept_keys_sorted]
        distillation_scores = [dist_sims[key] for key in concept_keys_sorted]
        x_labels = [generic_labels[key] for key in concept_keys_sorted]

        x = np.arange(len(x_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 6))

        add_color = '#3D5C6F'
        dist_color = '#E47159'

        rects1 = ax.bar(x - width/2, addition_scores, width, label='Addition Fusion', color=add_color, alpha=0.9, edgecolor='black', linewidth=0.5)
        rects2 = ax.bar(x + width/2, distillation_scores, width, label='Distillation Fusion', color=dist_color, alpha=0.9, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Concept Preservation (Cosine Similarity)', fontsize=14)
        ax.set_xlabel('Original Concept ID', fontsize=14)
        ax.set_title('Comparison of Concept Preservation by Fusion Method', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=12)
        ax.set_ylim(0, max(max(addition_scores, default=1), max(distillation_scores, default=1)) * 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        autolabel(rects1)
        autolabel(rects2)
        
        ax.legend(fontsize=8)
        fig.tight_layout()

        bar_chart_path = self.output_dir / "similarity_comparison_bar_chart.png"
        plt.savefig(bar_chart_path, dpi=300)
        plt.close()
        print(f"✓ Similarity bar chart saved to: {bar_chart_path}")
        
    def run_final_analysis(self):
        print(f"\n{'='*60}\n|| Starting Final Analysis ||\n{'='*60}")
        
        self.create_enhanced_pca_plot()
        
        similarity_results, concept_matching_data = self.calculate_comprehensive_similarity()
        
        if similarity_results:
            self.create_similarity_bar_chart(similarity_results)
            
        print(f"\n{'='*60}\n|| All outputs saved to: {self.output_dir.resolve()} ||\n{'='*60}")
        return similarity_results, concept_matching_data


def main():
    original_loras_dir = "/root/lora_weight_before_distill/distill_weight7"
    addition_path = "/root/lora_train/multi_combine_add/resistor_diode_light_addition.safetensors"
    distillation_path = "/root/lora_train/multi_combine_add/resistor_diode_light_distillation.safetensors"
    
    output_visualization_path = "lora_analysis_simplified_six_loras_final"

    PROMPTS = [
        "A photo of a great <resistor_front>.", "A photo of a great <resistor_up>.",
        "A photo of a great <diode_front>.", "A photo of a great <diode_up>.",
        "A photo of a great <light_diode_front>.", "A photo of a great <light_diode_up>."
    ]
    
    # SEEDS = [42, 1, 999, 0, 27]
    # SEEDS = [42]
    SEEDS = range(0,100)
    
    analyzer = LoRAOutputSimilarityAnalyzer(
        base_model_path=BASE_MODEL_PATH,
        output_dir_path=output_visualization_path
    )
    
    analyzer.load_multiple_original_loras(original_loras_dir)
    
    # 触发词列表应仅包含您实际希望融合模型响应的词
    # 即使模型是用多个概念训练的，您也可以只为分析中关心的概念提供触发词
    trigger_words_list = [
        "resistor_front", "resistor_up", "diode_front", "diode_up",
        "light_diode_front", "light_diode_up",
    ]
    
    analyzer.add_lora("LoRA_Addition_Fusion", addition_path, trigger_words=trigger_words_list, is_fusion=True)
    analyzer.add_lora("LoRA_Distillation_Fusion", distillation_path, trigger_words=trigger_words_list, is_fusion=True)
    
    analyzer.run_analysis(prompts=PROMPTS, seeds=SEEDS)
    similarity_results, concept_data = analyzer.run_final_analysis()
    
    print("\n--- Main script finished. Final results: ---")
    if similarity_results:
        summary_df = pd.DataFrame(similarity_results).T[['mean', 'std', 'min', 'max']]
        print(summary_df)
    
    return similarity_results, concept_data

if __name__ == "__main__":
    main()