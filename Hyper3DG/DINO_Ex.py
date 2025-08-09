import sys
sys.path.append("/home/s414e2/CJH/Text-to-3D/LucidDreamer")  # 添加项目根目录到Python路径

import torch
import torchvision.transforms as T
from . import vision_transformer as vits

class DINOFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用本地模型定义初始化DINO
        self.model = vits.__dict__["vit_small"](patch_size=8, num_classes=0)
        # 加载本地预训练权重
        state_dict = torch.load('/home/s414e2/CJH/Text-to-3D/LucidDreamer/Hyper3DG/dino_deitsmall8_pretrain.pth', map_location='cpu')
        self.model.load_state_dict(state_dict, strict=True)
        
        self.model.eval()
        self.model.to(self.device)
        
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
    def extract_features(self, patch_images):
        """
        从Tensor格式的patch渲染图像中提取DINO特征
        
        Args:
            patch_images (torch.Tensor): 输入图像张量 [B, C, H, W]，值范围[0,1]
            
        Returns:
            torch.Tensor: DINO特征
        """
        with torch.no_grad():
            # 直接处理Tensor输入
            batch = torch.stack([
                self.preprocess(image)  # image已经是Tensor
                for image in patch_images
            ]).to(self.device)
            
            # 提取特征
            outputs = self.model(batch)
            return outputs