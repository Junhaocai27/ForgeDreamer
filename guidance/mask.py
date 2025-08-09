import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import math

# --- 1. 初始设置和模型加载 ---
try:
    # 假设 'segment_anything' 在父目录
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("请确保 'segment_anything' 库已正确安装，并且路径已添加到 sys.path")
    sys.exit(1)

# --- 参数配置 ---
# sam_checkpoint = "/path/to/your/sam_vit_b_01ec64.pth" # 修改为你的模型路径
sam_checkpoint = "/root/LucidDreamer/guidance/sam_vit_b_01ec64.pth"
model_type = "vit_b"
# image_path = '/path/to/your/image.jpg' # 修改为你的图片路径
image_path = '/root/LucidDreamer/guidance/test_imgs/test_img.jpg'
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 策略选择 ---
# 'variance': 最小归一化方差策略 (你提供的代码，适合纯色背景)
# 'border': 边缘最大面积策略 (更通用，鲁棒性更强)
STRATEGY = 'border' 

# --- 图像和模型加载 ---
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    print(f"错误：无法读取图像 {image_path}")
    sys.exit(1)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
H, W, _ = image_rgb.shape

print("正在加载 SAM 模型...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
print("模型加载完成。")

# 我们可以调整 SAM 的参数以获得更好或更快的分割
# points_per_side: 在图像网格上采样的点数。增加可以得到更精细的分割，但会变慢。
# pred_iou_thresh: 预测的IOU阈值，用于过滤低质量的掩码。
# stability_score_thresh: 稳定性得分阈值，过滤不稳定的掩码。
# box_nms_thresh: 用于后处理去重的NMS阈值。
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # 过滤掉非常小的区域
)

# --- 2. 生成所有分割掩码 ---
print("正在生成分割掩码...")
masks = mask_generator.generate(image_rgb)
print(f"生成了 {len(masks)} 个分割块。")

if len(masks) < 1:
    print("错误：未能生成任何分割块。")
    sys.exit(1)

# --- 3. 应用策略选择主体 ---
final_binary_mask = np.zeros((H, W), dtype=bool)

if STRATEGY == 'variance':
    print("应用策略: 最小归一化方差 (适合纯色背景)")
    processed_masks = []
    for ann in masks:
        area = ann['area']
        mask = ann['segmentation']
        masked_pixels = image_rgb[mask]
        
        # 计算颜色标准差的均值
        color_std = np.std(masked_pixels, axis=0).mean()
        
        # 分数 = 标准差 / 面积的对数。惩罚颜色复杂或面积小的区域。
        # 加一个小的平滑项避免log(1)=0或log(0)错误。
        score = color_std / math.log(area + 1.01) 
        
        processed_masks.append({'ann': ann, 'score': score})

    if not processed_masks:
        print("警告：没有有效的掩码进行分析。")
        # 回退：选择最大面积块的反转作为主体
        background_ann = sorted(masks, key=lambda x: x['area'], reverse=True)[0]
        final_binary_mask = ~background_ann['segmentation']
    else:
        # 选择分数最低的掩码作为背景
        background = sorted(processed_masks, key=lambda x: x['score'])[0]
        print(f"识别到最可能的背景，其评分为: {background['score']:.4f}, 面积: {background['ann']['area']}")
        final_binary_mask = ~background['ann']['segmentation']

elif STRATEGY == 'border':
    print("应用策略: 边缘最大面积 (通用性强)")
    
    border_masks = []
    for ann in masks:
        mask = ann['segmentation']
        # 检查掩码是否与任何一边接触
        if (mask[0, :].any() or mask[-1, :].any() or 
            mask[:, 0].any() or mask[:, -1].any()):
            border_masks.append(ann)
    
    if not border_masks:
        print("警告：没有找到接触边缘的掩码。回退到'反转最大面积'策略。")
        background_ann = sorted(masks, key=lambda x: x['area'], reverse=True)[0]
        final_binary_mask = ~background_ann['segmentation']
    else:
        # 在所有接触边缘的掩码中，找到面积最大的那个作为背景
        background_ann = sorted(border_masks, key=lambda x: x['area'], reverse=True)[0]
        print(f"识别到最可能的背景，面积: {background_ann['area']}")
        final_binary_mask = ~background_ann['segmentation']

# --- 4. 保存和显示结果 ---
final_binary_mask_uint8 = final_binary_mask.astype(np.uint8) * 255

if final_binary_mask_uint8.sum() > 0:
    output_suffix = f"strategy_{STRATEGY}"
    output_mask_path = f"subject_binary_mask_{output_suffix}.png"
    cv2.imwrite(output_mask_path, final_binary_mask_uint8)
    print(f"最终的二值掩码已保存为: {output_mask_path}")

    # 创建带透明通道的前景图像
    foreground_img = np.zeros((H, W, 4), dtype=np.uint8)
    foreground_img[:, :, :3] = image_rgb
    foreground_img[:, :, 3] = final_binary_mask_uint8
    
    output_foreground_path = f"subject_foreground_{output_suffix}.png"
    # 保存时需要从 RGBA 转换回 BGRA 给 OpenCV
    cv2.imwrite(output_foreground_path, cv2.cvtColor(foreground_img, cv2.COLOR_RGBA2BGRA))
    print(f"分割出的主体已保存为透明PNG: {output_foreground_path}")

    # 可视化
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("原始图像")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(final_binary_mask_uint8, cmap='gray')
    plt.title(f"最终主体掩码 (策略: {STRATEGY})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(foreground_img)
    plt.title("提取的主体")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print("未能生成有效的最终掩码。")