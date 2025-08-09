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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
import json
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
            self.pipe = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            print("Successfully loaded model and set EulerAncestralDiscreteScheduler")
        except Exception as e:
            raise RuntimeError(f"Failed to load the base model, please check the path: {base_model_path}\nError: {e}")

        print(f"Loading image feature extractor: {CLIP_MODEL_ID}")
        self.feature_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
        self.feature_extractor = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(self.device)
        self.feature_extractor.eval()
        
        self.original_loras: Dict[str, Path] = {}
        self.fusion_loras: Dict[str, Path] = {}
        # 核心改动：存储每个LoRA下每个概念的特征
        # 结构: {'lora_name': {'concept_name': np.ndarray}}
        self.concept_features: Dict[str, Dict[str, np.ndarray]] = {}
        self.lora_triggers: Dict[str, List[str]] = {}
        self.generated_samples: Dict[str, List[Image.Image]] = {}
        
        self.output_dir = Path(output_dir_path)
        self.output_dir.mkdir(exist_ok=True)
        print(f"All analysis results will be saved to: {self.output_dir.resolve()}")

        # 使用更适合中文显示的字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

    def add_lora(self, name: str, path: str, trigger_words: List[str] = None, is_fusion: bool = False):
        lora_path = Path(path)
        if not lora_path.exists():
            warnings.warn(f"Warning: LoRA file does not exist, skipped: {path}")
            return
        # 对于原始LoRA，其触发词就是文件名
        if trigger_words is None and not is_fusion:
            trigger_words = [lora_path.stem.lower()]
        
        self.lora_triggers[name] = trigger_words
        if is_fusion:
            self.fusion_loras[name] = lora_path
            print(f"Registered fusion LoRA: {name}, supported triggers: {trigger_words}")
        else:
            self.original_loras[name] = lora_path
            print(f"Registered original LoRA: {name}, trigger: {trigger_words}")

    def _extract_triggers_from_filename(self, filename: str) -> List[str]:
        return [filename.lower()]

    def load_multiple_original_loras(self, lora_dir: str, pattern: str = "*.safetensors", trigger_mapping: Dict[str, List[str]] = None):
        lora_path = Path(lora_dir)
        for f in lora_path.glob(pattern):
            manual_triggers = None
            if trigger_mapping and f.stem in trigger_mapping:
                manual_triggers = trigger_mapping[f.stem]
            # 默认is_fusion=False，触发词从文件名中提取
            self.add_lora(f.stem, str(f), trigger_words=manual_triggers, is_fusion=False)
            
    # _load_lora_with_patch_pipe, _extract_trigger_from_prompt, _reset_pipe_to_original 方法保持不变
    def _load_lora_with_patch_pipe(self, lora_path: Path):
        try:
            from lora_diffusion import tune_lora_scale, patch_pipe
            print(f"    Loading LoRA with new process: {lora_path}")
            pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16).to(self.device)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            patch_pipe(pipe, str(lora_path), patch_text=True, patch_ti=True, patch_unet=True)
            tune_lora_scale(pipe.unet, 1.00)
            tune_lora_scale(pipe.text_encoder, 1.00)
            self.pipe = pipe
            return True
        except ImportError:
            warnings.warn("The lora_diffusion library is not installed, cannot use the patch_pipe method.")
            return False
        except Exception as e:
            warnings.warn(f"Failed to load LoRA using patch_pipe: {e}")
            return False

    def _extract_trigger_from_prompt(self, prompt: str) -> str:
        match = re.search(r'<([^>]+)>', prompt)
        return match.group(1).lower() if match else ""
        
    def _reset_pipe_to_original(self):
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16).to(self.device)
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        except Exception as e:
            warnings.warn(f"Failed to reset pipeline: {e}")
            
    # _find_matching_loras 逻辑需要调整，确保一个prompt只对应一个原始LoRA，但可以对应所有融合LoRA
    def _find_matching_loras(self, prompt: str) -> List[str]:
        trigger_in_prompt = self._extract_trigger_from_prompt(prompt)
        if not trigger_in_prompt: 
            return []
        
        matching_loras = []
        print(f"  Finding matching LoRA for trigger: '{trigger_in_prompt}'")

        # 1. 查找匹配的原始LoRA (精确匹配)
        for lora_name, triggers in self.lora_triggers.items():
            if lora_name in self.original_loras and trigger_in_prompt in triggers:
                print(f"    Exact match found for original LoRA: '{lora_name}'")
                matching_loras.append(lora_name)

        # 2. 为所有融合LoRA添加此任务（只要prompt中的触发词在它们支持的列表里）
        for lora_name, triggers in self.lora_triggers.items():
            if lora_name in self.fusion_loras and trigger_in_prompt in triggers:
                print(f"    Trigger supported by fusion LoRA: '{lora_name}'")
                matching_loras.append(lora_name)
                
        print(f"  Found matching LoRAs for this prompt: {matching_loras}")
        return list(set(matching_loras)) # 去重

    @torch.no_grad()
    def generate_and_extract_features(self, lora_name: str, lora_path: Path, prompt: str, seed: int, save_sample: bool = True) -> np.ndarray:
        # 这个方法基本不变，只是调用它的地方会改变
        print(f"  > Processing: LoRA='{lora_name}', Prompt='{prompt}', Seed={seed}")
        try:
            success = self._load_lora_with_patch_pipe(lora_path)
            if not success:
                self._reset_pipe_to_original()
                self.pipe.load_lora_weights(lora_path.parent, weight_name=lora_path.name)
            
            generator = torch.Generator(self.device).manual_seed(seed)
            image: Image.Image = self.pipe(prompt, num_inference_steps=50, generator=generator, guidance_scale=7.5).images[0]
            
            # 保存样本图片的逻辑可以保持
            if save_sample:
                concept_name = self._extract_trigger_from_prompt(prompt)
                sample_key = f"{lora_name}_{concept_name}"
                if sample_key not in self.generated_samples: 
                    self.generated_samples[sample_key] = []
                if len(self.generated_samples[sample_key]) < 3: 
                    self.generated_samples[sample_key].append(image.copy())
            
            inputs = self.feature_processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.feature_extractor(**inputs).image_embeds
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            warnings.warn(f"An error occurred while processing LoRA '{lora_name}' with prompt '{prompt}': {e}")
            return np.array([])

    def run_analysis(self, prompts: List[str], seeds: List[int]):
        print("\n--- Starting Concept-Based Similarity Analysis ---")
        
        # 临时存储所有特征，结构: {'lora_name': {'concept_name': [features_from_seeds]}}
        temp_concept_features = {}
        
        all_loras = {**self.original_loras, **self.fusion_loras}

        for prompt in prompts:
            concept_name = self._extract_trigger_from_prompt(prompt)
            if not concept_name:
                continue

            matching_loras = self._find_matching_loras(prompt)
            
            for lora_name in matching_loras:
                if lora_name not in all_loras: 
                    continue
                lora_path = all_loras[lora_name]

                # 初始化字典
                if lora_name not in temp_concept_features:
                    temp_concept_features[lora_name] = {}
                if concept_name not in temp_concept_features[lora_name]:
                    temp_concept_features[lora_name][concept_name] = []
                
                # 对每个seed生成图像并提取特征
                for seed in seeds:
                    features = self.generate_and_extract_features(lora_name, lora_path, prompt, seed)
                    if features.size > 0:
                        temp_concept_features[lora_name][concept_name].append(features)

        # 计算每个(LoRA, Concept)组合的平均特征向量
        print("\n--- Averaging features for each concept ---")
        for lora_name, concepts in temp_concept_features.items():
            if lora_name not in self.concept_features:
                self.concept_features[lora_name] = {}
            for concept_name, features_list in concepts.items():
                if features_list:
                    avg_features = np.mean(features_list, axis=0)
                    self.concept_features[lora_name][concept_name] = avg_features
                    print(f"  - Calculated average feature for ({lora_name}, {concept_name}) from {len(features_list)} samples.")
                else:
                    warnings.warn(f"    Warning: No features generated for ({lora_name}, {concept_name}).")
        
        print("--- Analysis complete ---")

    # 旧的create_feature_tsne_plot将被新的方法替代
    def create_feature_tsne_plot(self):
        """
        新的t-SNE绘图入口，调用新的绘图函数。
        """
        print("\n--- Generating Concept-Based t-SNE Visualization ---")
        self._plot_concept_clusters()

    def _plot_concept_clusters(self):
        """
        根据“概念”来组织和绘制t-SNE图，使用不同颜色代表概念，不同形状代表模型类型。
        """
        # 1. 准备数据框 (DataFrame)
        plot_data = []
        for lora_name, concepts in self.concept_features.items():
            for concept_name, feature_vector in concepts.items():
                if lora_name in self.original_loras:
                    model_type = "原始LoRA"
                elif "Addition" in lora_name:
                    model_type = "加法融合"
                elif "Distillation" in lora_name:
                    model_type = "蒸馏融合"
                else:
                    model_type = "其他"

                plot_data.append({
                    "lora": lora_name,
                    "concept": concept_name,
                    "model_type": model_type,
                    "feature": feature_vector
                })
        
        if not plot_data:
            print("No concept features available to plot.")
            return

        df = pd.DataFrame(plot_data)
        features = np.array(df["feature"].tolist())
        
        if len(df) < 3:
            print("Warning: Need at least 3 data points for t-SNE. Skipping plot.")
            return

        # L2 标准化
        features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)

        # 2. t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) - 1), 
                    learning_rate='auto', init='pca', n_iter=2000)
        df[['tsne1', 'tsne2']] = tsne.fit_transform(features_normalized)

        # 3. 绘图
        plt.figure(figsize=(16, 12))
        
        # 定义颜色和形状
        concepts = df['concept'].unique()
        model_types = df['model_type'].unique()
        
        # 使用更鲜明的颜色和形状
        colors = plt.cm.get_cmap('tab10', len(concepts))
        markers = ['o', 's', 'X', '^']
        
        color_map = {concept: colors(i) for i, concept in enumerate(concepts)}
        marker_map = {mtype: markers[i] for i, mtype in enumerate(model_types)}
        
        # 使用seaborn进行绘制，更简洁
        sns.scatterplot(
            data=df,
            x='tsne1',
            y='tsne2',
            hue='concept',
            style='model_type',
            palette=color_map,
            markers=marker_map,
            s=200,      # 增大点的大小
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )

        # 可选：绘制连接线，连接同一概念的不同模型点
        for concept_name, group in df.groupby('concept'):
            if len(group) > 1:
                # 找到原始点
                original_point = group[group['model_type'] == '原始LoRA']
                if not original_point.empty:
                    ox, oy = original_point.iloc[0]['tsne1'], original_point.iloc[0]['tsne2']
                    # 连接到其他点
                    for _, row in group[group['model_type'] != '原始LoRA'].iterrows():
                        plt.plot([ox, row['tsne1']], [oy, row['tsne2']], 
                                 color=color_map[concept_name], linestyle='--', alpha=0.6, linewidth=1.5)

        plt.title('t-SNE 可视化：按概念聚类', fontsize=20, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        plt.legend(title='图例', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局为图例留出空间

        tsne_path = self.output_dir / "concept_cluster_tsne.png"
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Concept-based t-SNE plot saved to: {tsne_path}")
        
    # 其他可视化方法（如热力图）可能需要调整或暂时禁用，因为它们基于旧的数据结构
    def create_similarity_heatmap(self):
        print("\n--- Similarity Heatmap (Skipped) ---")
        print("Note: The heatmap is based on model-level features and is not suitable for the new concept-based analysis.")
        return

    def _analyze_high_dimensional_centroid(self):
        print("\n--- High-Dimensional Centroid Analysis (Skipped) ---")
        print("Note: Centroid analysis needs to be adapted for the new concept-based structure.")
        return
        
    def save_sample_images(self):
        sample_dir = self.output_dir / "sample_images_by_concept"
        sample_dir.mkdir(exist_ok=True)
        print(f"\n--- Saving sample images to {sample_dir} ---")
        for sample_key, images in self.generated_samples.items():
            # sample_key is "lora_name_concept_name"
            lora_dir = sample_dir / sample_key.replace(" ", "_")
            lora_dir.mkdir(exist_ok=True)
            for i, img in enumerate(images):
                # 种子信息丢失了，但可以按顺序保存
                img_path = lora_dir / f"sample_{i+1}.png"
                img.save(img_path)
                print(f"Saved: {img_path}")


    def generate_all_visualizations(self):
        print(f"\n{'='*60}\n|| Starting Visualization Generation ||\n{'='*60}")
        try:
            self.save_sample_images()
            # self.create_similarity_heatmap()  # 暂时跳过
            self.create_feature_tsne_plot()  # 使用新的t-SNE
            print(f"\n{'='*60}\n|| All visualizations have been saved to: {self.output_dir.resolve()} ||\n{'='*60}")
        except Exception as e:
            print(f"An error occurred during visualization generation: {e}")
            warnings.warn("Some visualizations may not have been generated correctly.")

def main():
    original_loras_dir = "/root/lora_weight_before_distill/distill_weight5"
    addition_path = "/root/lora_train/multi_combine_add/resistor_diode_addition.safetensors"
    distillation_path = "/root/lora_train/multi_combine_add/resistor_diode_distillation.safetensors"
    
    output_visualization_path = "lora_analysis_results_4_loras"

    # Prompts现在驱动了整个分析的核心
    PROMPTS = [
        "A photo of a great <resistor_front>.",
        "A photo of a great <resistor_up>.",
        "A photo of a great <diode_front>.",
        "A photo of a great <diode_up>.",
    ]
    SEEDS = range(1,101) # 使用多个seed来获得更稳定的平均特征
    
    try:
        analyzer = LoRAOutputSimilarityAnalyzer(
            base_model_path=BASE_MODEL_PATH,
            output_dir_path=output_visualization_path
        )
        
        # 1. 加载原始LoRA
        analyzer.load_multiple_original_loras(original_loras_dir)
        
        # 2. 加载融合LoRA，并明确告知它们支持哪些触发词
        # 这里的触发词列表至关重要，决定了融合模型会参与哪些概念的生成
        supported_triggers = ["resistor_front", "resistor_up", "diode_front", "diode_up", "light_diode_front", "light_diode_up"]
        
        analyzer.add_lora(
            "LoRA_Addition_Fusion", 
            addition_path, 
            trigger_words=supported_triggers, 
            is_fusion=True
        )
        analyzer.add_lora(
            "LoRA_Distillation_Fusion", 
            distillation_path, 
            trigger_words=supported_triggers, 
            is_fusion=True
        )
        
        # 3. 运行分析
        analyzer.run_analysis(prompts=PROMPTS, seeds=SEEDS)
        
        # 4. 生成可视化结果
        analyzer.generate_all_visualizations()

    except Exception as e:
        print(f"\nA critical error occurred in the program: {e}")
        import traceback
        traceback.print_exc()
        print("Please check:\n1. If PyTorch, CUDA, Diffusers, and lora-diffusion are installed correctly.\n2. If the base model path is correct.\n3. If there is sufficient VRAM on the server.")

if __name__ == "__main__":
    main()