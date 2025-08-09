import os
import sys
import math
import itertools
from functools import partial
import urllib.request
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import mmcv
from mmcv.runner import load_checkpoint

# 确保可以导入mmseg和dinov2
try:
    from mmseg.apis import init_segmentor, inference_segmentor
except ImportError:
    print("[WARNING] mmseg未安装，尝试安装...")
    from mmseg.apis import init_segmentor, inference_segmentor

# 设置DINOv2路径
DINOV2_PATH = '/root/LucidDreamer/dinov2'
if not os.path.exists(DINOV2_PATH):
    print(f"[ERROR] 找不到DINOv2路径: {DINOV2_PATH}")
    print("请确认DINOv2已正确安装，或修改DINOV2_PATH变量")
    sys.exit(1)

# 将DINOv2路径添加到系统路径
sys.path.append(DINOV2_PATH)

# 导入DINOv2相关模块
try:
    import dinov2.eval.segmentation.models
    import dinov2.eval.segmentation.utils.colormaps as colormaps
    import dinov2.eval.segmentation_m2f.models.segmentors
except ImportError:
    print("[ERROR] 无法导入DINOv2模块，请检查DINOv2安装")
    sys.exit(1)

# 创建输出目录
OUTPUT_DIR = "/root/LucidDreamer/guidance/enhanced_segmentation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输入图像路径
IMAGE_PATH = "/root/LucidDreamer/guidance/test_img.jpg"

# 模型配置
BACKBONE_SIZE = "giant"  # 可选: small, base, large, giant
HEAD_DATASET = "ade20k"  # 可选: ade20k, voc2012
HEAD_TYPE = "ms"         # 可选: ms, linear
HEAD_SCALE_COUNT = 3     # 更多尺度=更慢但更精确 (1-5)

# 数据集颜色映射
DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}

# 辅助类和函数
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple
    
    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right
    
    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

def create_segmenter(cfg, backbone_model):
    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0])
        )
    model.init_weights()
    return model

def load_config_from_url(url: str) -> str:
    print(f"[INFO] 从URL加载配置: {url}")
    try:
        with urllib.request.urlopen(url) as f:
            return f.read().decode()
    except Exception as e:
        print(f"[ERROR] 加载配置失败: {e}")
        sys.exit(1)

def render_segmentation(segmentation_logits, dataset):
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)

# 主程序
def main():
    print(f"[INFO] 开始处理图像: {IMAGE_PATH}")
    
    # 加载图像
    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
        print(f"[INFO] 成功加载图像，尺寸: {image.size}")
    except Exception as e:
        print(f"[ERROR] 加载图像失败: {e}")
        return
    
    # 设置骨干网络
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"
    print(f"[INFO] 使用模型: {backbone_name}")
    
    # 加载骨干网络
    print("[INFO] 加载DINOv2骨干网络...")
    try:
        # 使用本地路径加载模型
        torch.hub.set_dir(DINOV2_PATH)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", 
                                        model=backbone_name,
                                        source='local',
                                        pretrained=False)
        backbone_model.eval()
        backbone_model.to(device)
        print(f"[INFO] 骨干网络加载成功，使用设备: {device}")
    except Exception as e:
        print(f"[ERROR] 加载骨干网络失败: {e}")
        return
    
    # 配置URL
    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"
    
    # 加载配置
    print(f"[INFO] 加载分割头配置...")
    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    
    # 调整尺度
    if HEAD_TYPE == "ms":
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
        print(f"[INFO] 使用尺度: {cfg.data.test.pipeline[1]['img_ratios']}")
    
    # 创建分割器并加载权重
    print("[INFO] 创建分割器并加载权重...")
    model = create_segmenter(cfg, backbone_model=backbone_model)
    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.to(device)
    model.eval()
    
    # 进行推理
    print("[INFO] 执行分割推理...")
    array = np.array(image)[:, :, ::-1]  # RGB转BGR
    segmentation_logits = inference_segmentor(model, array)[0]
    segmented_image = render_segmentation(segmentation_logits, HEAD_DATASET)
    
    # 保存结果
    segmented_path = os.path.join(OUTPUT_DIR, "segmentation_result.png")
    segmented_image.save(segmented_path)
    print(f"[INFO] 分割结果保存至: {segmented_path}")
    
    # 保存原图与分割结果对比
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("原始图像")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("分割结果")
    plt.axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, "segmentation_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    print(f"[INFO] 对比图保存至: {comparison_path}")
    
    # 创建二值掩码
    # 提取人物/前景类别 (此处示例，实际类别ID需要根据数据集调整)
    if HEAD_DATASET == "ade20k":
        # ADE20K中的人物类别ID (根据需要调整)
        foreground_ids = [10]  # 10是人物类别ID
    else:  # VOC2012
        foreground_ids = [15]  # 15是人物类别ID
    
    mask = np.zeros_like(segmentation_logits, dtype=np.uint8)
    for idx in foreground_ids:
        mask[segmentation_logits == idx] = 255
    
    # 保存掩码
    mask_img = Image.fromarray(mask)
    mask_path = os.path.join(OUTPUT_DIR, "mask.png")
    mask_img.save(mask_path)
    print(f"[INFO] 二值掩码保存至: {mask_path}")
    
    print("[INFO] 处理完成!")

if __name__ == "__main__":
    main()