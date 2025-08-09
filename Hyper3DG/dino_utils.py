import torch
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel

class DINOAttentionMask:
    def __init__(self, device='cuda'):
        """初始化DINO特征提取器"""
        self.device = device
        # 加载预训练的ViT模型和特征提取器
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits8')
        self.model = ViTModel.from_pretrained('facebook/dino-vits8', output_attentions=True).to(device)
        self.model.eval()
        
    def extract_attention_maps(self, image, layer_idx=-1):
        """
        提取DINO模型的注意力图
        Args:
            image: 输入图像 (C,H,W)，值范围[0,1]
            layer_idx: 提取哪一层的注意力 (默认最后一层)
        Returns:
            attention_map: 归一化的注意力图 (H,W)
        """
        # 将图像转换为PIL格式并应用DINO预处理
        img_tensor = image.cpu().permute(1, 2, 0).numpy() * 255
        img_tensor = img_tensor.astype('uint8')
        
        # 提取特征并获取注意力
        with torch.no_grad():
            inputs = self.feature_extractor(images=img_tensor, return_tensors="pt")
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            
            # 获取最后一层注意力
            attentions = outputs.attentions[layer_idx]  # [1, num_heads, seq_len, seq_len]
            
            # 使用[CLS]令牌的注意力作为特征图
            cls_attn = attentions[0, :, 0, 1:].mean(0)  # 平均所有头的注意力
            
            # 重塑为特征图 (去掉CLS令牌)
            patch_size = 8  # ViT-S/8的patch大小
            num_patches = int(cls_attn.shape[0] ** 0.5)
            attention_map = cls_attn.reshape(num_patches, num_patches)
            
            # 上采样到原始图像大小
            h, w = image.shape[1], image.shape[2]
            attention_map = F.interpolate(
                attention_map.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bicubic',
                align_corners=False
            ).squeeze()
            
            # 归一化注意力图
            min_val = attention_map.min()
            max_val = attention_map.max()
            attention_map = (attention_map - min_val) / (max_val - min_val + 1e-8)
            
        return attention_map
    
    def generate_masks(self, image, k_ratio=0.2, soft_scale=0.5):
        """
        从图像中生成二值掩码和软掩码
        Args:
            image: 输入图像 (C,H,W)，值范围[0,1]
            k_ratio: topk的比例
            soft_scale: 软掩码的缩放系数
        Returns:
            binary_mask: 二值掩码 (H,W)
            combined_mask: 结合掩码 (H,W)
        """
        # 提取注意力图
        attention_map = self.extract_attention_maps(image)
        h, w = attention_map.shape
        
        # 计算topk的阈值
        k = int(h * w * k_ratio)
        values, _ = torch.topk(attention_map.flatten(), k)
        threshold = values[-1]
        
        # 生成二值掩码（topk部分为1）
        binary_mask = (attention_map >= threshold).float()
        
        # 生成软掩码（非topk部分按原始值缩放）
        soft_mask = attention_map * (1 - binary_mask) * soft_scale
        
        # 合并掩码
        combined_mask = binary_mask + soft_mask
        
        return binary_mask, combined_mask
    
    def apply_mask_to_image(self, image, combined_mask):
        """
        将掩码应用到图像上
        Args:
            image: 输入图像 (C,H,W)
            combined_mask: 结合的掩码 (H,W)
        Returns:
            masked_image: 掩码增强后的图像 (C,H,W)
        """
        # 应用掩码到每个通道
        masked_image = image * combined_mask.unsqueeze(0)
        return masked_image