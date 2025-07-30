import torch
from torchvision import transforms
from PIL import Image
import os
from torch import nn

def load_dino_model(weights_path):
    """
    从本地权重文件加载DINO模型
    
    Args:
        weights_path: DINO模型权重文件的路径
    
    Returns:
        加载好的DINO模型
    """
    # 确保权重文件存在
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"DINO权重文件未找到: {weights_path}")
    
    # 创建DINO ViT-S/8模型结构
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', pretrained=False)
    
    # 加载预训练权重
    state_dict = torch.load(weights_path, map_location="cpu")
    # 有些权重文件可能包含额外的键，如果需要，可以过滤掉
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

def preprocess_tensor(image_tensor):
    """
    预处理tensor格式的图像以适应DINO模型的输入要求
    
    Args:
        image_tensor: 输入图像tensor，形状为[C, H, W]或[B, C, H, W]
    
    Returns:
        预处理后的图像tensor，形状为[B, C, H, W]
    """
    # 确保输入是4D tensor [B, C, H, W]
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
    
    # 确保尺寸正确
    if image_tensor.shape[2] != 224 or image_tensor.shape[3] != 224:
        resize = transforms.Resize((224, 224))
        image_tensor = resize(image_tensor)
    
    # 归一化图像
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensor = normalize(image_tensor)
    
    return image_tensor

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def extract_features_from_tensor(image_tensor, model):
    """
    从tensor格式的图像中提取特征
    
    Args:
        image_tensor: 输入图像tensor
        model: DINO模型
    
    Returns:
        提取的特征
    """
    processed_tensor = preprocess_tensor(image_tensor)
    with torch.no_grad():
        features = model(processed_tensor)
    return features

def extract_features(image_path, model):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = model(image_tensor)
    return features

# Example usage:
# model = load_dino_model('/path/to/dino_weights.pth')
# image_tensor = torch.randn(3, 256, 256)  # 示例tensor图像
# features = extract_features_from_tensor(image_tensor, model)
# print(features)
