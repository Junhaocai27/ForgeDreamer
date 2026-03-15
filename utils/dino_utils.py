import torch
from torchvision import transforms
from PIL import Image
import os
from torch import nn

def load_dino_model(weights_path):
    """
    Load the DINO model from a local weights file
    
    Args:
        weights_path: path to the DINO model weights file
    
    Returns:
        the loaded DINO model
    """
    # Ensure the weights file exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"DINO weights file not found: {weights_path}")
    
    # Create the DINO ViT-S/8 model architecture
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', pretrained=False)
    
    # Load pretrained weights
    state_dict = torch.load(weights_path, map_location="cpu")
    # Some weight files may contain extra keys; filter them out if needed
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

def preprocess_tensor(image_tensor):
    """
    Preprocess an image tensor to match DINO model input requirements
    
    Args:
        image_tensor: input image tensor with shape [C, H, W] or [B, C, H, W]
    
    Returns:
        preprocessed image tensor with shape [B, C, H, W]
    """
    # Ensure input is a 4D tensor [B, C, H, W]
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)  # add batch dimension
    
    # Ensure correct size
    if image_tensor.shape[2] != 224 or image_tensor.shape[3] != 224:
        resize = transforms.Resize((224, 224))
        image_tensor = resize(image_tensor)
    
    # Normalize the image
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
    Extract features from an image tensor
    
    Args:
        image_tensor: input image tensor
        model: DINO model
    
    Returns:
        extracted features
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
# image_tensor = torch.randn(3, 256, 256)  # example tensor image
# features = extract_features_from_tensor(image_tensor, model)
# print(features)
