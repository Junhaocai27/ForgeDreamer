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
import pandas as pd
import warnings
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
            self.pipe = self.pipe.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the base model: {base_model_path}\nError: {e}")

        self.feature_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
        self.feature_extractor = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(self.device)
        self.feature_extractor.eval()
        
        self.original_loras: Dict[str, Path] = {}
        self.fusion_loras: Dict[str, Path] = {}
        # NEW: Storing features per (LoRA, Concept) pair
        self.concept_features: Dict[Tuple[str, str], np.ndarray] = {}
        self.lora_triggers: Dict[str, List[str]] = {}
        
        self.output_dir = Path(output_dir_path)
        self.output_dir.mkdir(exist_ok=True)
        print(f"Analysis results will be saved to: {self.output_dir.resolve()}")

        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

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
            pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16).to(self.device)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            patch_pipe(pipe, str(lora_path), patch_text=True, patch_ti=True, patch_unet=True)
            tune_lora_scale(pipe.unet, 1.00); tune_lora_scale(pipe.text_encoder, 1.00)
            self.pipe = pipe
            return True
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
            if not trigger: continue

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

    def create_feature_pca_plot(self):
        if len(self.concept_features) < 2:
            print(f"PCA Plot SKIPPED: At least 2 concept points are required, but found {len(self.concept_features)}.")
            return

        print("\n--- Generating Disaggregated Concept PCA Visualization ---")
        
        concept_keys = list(self.concept_features.keys())
        features = np.array(list(self.concept_features.values()))
        features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features_normalized)
        coords_map = {key: features_2d[i] for i, key in enumerate(concept_keys)}

        plt.figure(figsize=(16, 12))
        plotted_labels = set()

        # Plot all the points first
        for key, coords in coords_map.items():
            lora_name, trigger_word = key
            
            if lora_name == "LoRA_Addition_Fusion":
                style = {'c': 'green', 'marker': 'P', 's': 250, 'label': 'Addition-Generated Concept'}
            elif lora_name == "LoRA_Distillation_Fusion":
                style = {'c': 'red', 'marker': 'X', 's': 250, 'label': 'Distillation-Generated Concept'}
            else:
                style = {'c': 'blue', 'marker': 'o', 's': 250, 'label': 'Original Concept'}

            label = style.pop('label')
            if label not in plotted_labels:
                plt.scatter(coords[0], coords[1], label=label, **style, edgecolors='black', alpha=0.9, zorder=5)
                plotted_labels.add(label)
            else:
                plt.scatter(coords[0], coords[1], **style, edgecolors='black', alpha=0.9, zorder=5)
            
            plt.annotate(trigger_word, (coords[0], coords[1]), xytext=(8, 8), textcoords='offset points', fontsize=12, zorder=6)
        
        # Draw connecting lines
        unique_triggers = sorted(list(set(k[1] for k in coords_map.keys())))
        for trigger in unique_triggers:
            try:
                # Find the key for the original concept (lora name matches trigger word)
                original_key = next(k for k in coords_map if k[0] == trigger)
                original_coords = coords_map[original_key]

                # Connect Addition point to original
                addition_key = ("LoRA_Addition_Fusion", trigger)
                if addition_key in coords_map:
                    plt.plot([coords_map[addition_key][0], original_coords[0]],
                             [coords_map[addition_key][1], original_coords[1]],
                             color='green', linestyle='--', linewidth=2, alpha=0.8, zorder=1)

                # Connect Distillation point to original
                distillation_key = ("LoRA_Distillation_Fusion", trigger)
                if distillation_key in coords_map:
                    plt.plot([coords_map[distillation_key][0], original_coords[0]],
                             [coords_map[distillation_key][1], original_coords[1]],
                             color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=1)
            except StopIteration:
                warnings.warn(f"Could not find original LoRA for trigger '{trigger}' to draw lines.")

        plt.title('Disaggregated PCA: Original vs. Fusion-Generated Concepts', fontsize=18, fontweight='bold')
        plt.xlabel('PCA Component 1', fontsize=14)
        plt.ylabel('PCA Component 2', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.5)
        plt.tight_layout()

        pca_path = self.output_dir / "feature_pca_disaggregated_plot.png"
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Disaggregated PCA plot saved to: {pca_path}")

    def generate_all_visualizations(self):
        print(f"\n{'='*60}\n|| Starting Visualizations ||\n{'='*60}")
        self.create_feature_pca_plot()
        print(f"\n{'='*60}\n|| Visualizations saved to: {self.output_dir.resolve()} ||\n{'='*60}")

def main():
    original_loras_dir = "/root/lora_weight_before_distill/distill_weight7"
    addition_path = "/root/lora_train/multi_combine_add/resistor_diode_light_addition.safetensors"
    distillation_path = "/root/lora_train/multi_combine_add/resistor_diode_light_distillation.safetensors"
    
    output_visualization_path = "lora_analysis_disaggregated"

    # This prompt list will generate 6 distinct points for the PCA plot:
    # 2 from original LoRAs, 2 from Addition, 2 from Distillation.
    PROMPTS = [
        "A photo of a great <resistor_front>.",
        "A photo of a great <resistor_up>.",
        "A photo of a great <diode_front>.",
        "A photo of a great <diode_up>.",
        "A photo of a great <light_diode_front>.",
        "A photo of a great <light_diode_up>.",
    ]
    SEEDS = [42] # Using multiple seeds for more stable results
    
    analyzer = LoRAOutputSimilarityAnalyzer(
        base_model_path=BASE_MODEL_PATH,
        output_dir_path=output_visualization_path
    )
    
    # Load original LoRAs (e.g., resistor_front.safetensors, resistor_up.safetensors)
    analyzer.load_multiple_original_loras(original_loras_dir)
    
    # Register the fusion LoRAs
    analyzer.add_lora(
        "LoRA_Addition_Fusion", 
        addition_path, 
        trigger_words=["resistor_front", "resistor_up", "diode_front", "diode_up", "light_diode_front", "light_diode_up"], 
        is_fusion=True
    )
    analyzer.add_lora(
        "LoRA_Distillation_Fusion", 
        distillation_path, 
        trigger_words=["resistor_front", "resistor_up", "diode_front", "diode_up", "light_diode_front", "light_diode_up"], 
        is_fusion=True
    )
    
    analyzer.run_analysis(prompts=PROMPTS, seeds=SEEDS)
    analyzer.generate_all_visualizations()

if __name__ == "__main__":
    main()