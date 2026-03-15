import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from fastsam import FastSAM, FastSAMPrompt
import os


class SubjectMaskGenerator:
    """
    Class for subject mask generation using FastSAM
    """
    def __init__(self, device, fastsam_checkpoint_path="FastSAM/FastSAM.pt"):
        """
        Initialize the FastSAM model
        
        Args:
            device: compute device (cuda/cpu)
            fastsam_checkpoint_path: path to the FastSAM model weights
        """
        self.device = device
        self.fastsam = None
        self.mask_generator = None
        
        print("[MaskGenerator] Initializing FastSAM...")
        try:
            # Load the FastSAM model
            if os.path.exists(fastsam_checkpoint_path):
                self.fastsam = FastSAM(fastsam_checkpoint_path)
                self.mask_generator = self.fastsam  # maintain compatibility
                print(f"[MaskGenerator] FastSAM model loaded successfully from: {fastsam_checkpoint_path}")
            else:
                print(f"[MaskGenerator] FATAL: FastSAM checkpoint not found at: {fastsam_checkpoint_path}")
                
        except Exception as e:
            print(f"[MaskGenerator] FATAL: Failed to load FastSAM model from {fastsam_checkpoint_path}. Error: {e}")
            # The calling code should handle the case where self.mask_generator is None
    
    # def generate_mask_from_np(self, image_rgb_np, strategy='border', **kwargs):
    #     """
    #     Generate a subject binary mask from a NumPy RGB image array
        
    #     Args:
    #         image_rgb_np (np.ndarray): NumPy format RGB image, shape [H, W, 3], dtype uint8
    #         strategy (str): background recognition strategy, supports 'border', 'variance', 'largest', 'center'
            
    #     Returns:
    #         np.ndarray: boolean mask of shape [H, W], True indicates subject, None on failure
    #     """
    #     if self.mask_generator is None:
    #         print("[MaskGenerator] ERROR: FastSAM model not available for mask generation.")
    #         return None

    #     print(f"[MaskGenerator] Generating segments for a {image_rgb_np.shape} image using FastSAM...")
        
    #     try:
    #         # Convert to PIL Image
    #         pil_image = Image.fromarray(image_rgb_np).convert("RGB")
            
    #         # FastSAM inference
    #         everything_results = self.fastsam(
    #             pil_image,
    #             device=self.device,
    #             retina_masks=True,
    #             imgsz=1024,
    #             conf=0.4,
    #             iou=0.9
    #         )
            
    #         # Create prompt processor and get all segmentation results
    #         prompt_process = FastSAMPrompt(pil_image, everything_results, device=self.device)
    #         masks_tensor = prompt_process.everything_prompt()
            
    #         if masks_tensor is None or masks_tensor.shape[0] == 0:
    #             print("[MaskGenerator] WARN: FastSAM did not produce any segments.")
    #             return None
            
    #         # Convert to numpy format and create SAM-like mask structures
    #         masks = []
    #         for i in range(masks_tensor.shape[0]):
    #             mask_np = masks_tensor[i].cpu().numpy().astype(bool)
    #             area = np.sum(mask_np)
    #             if area > 0:  # keep non-empty masks only
    #                 masks.append({
    #                     'segmentation': mask_np,
    #                     'area': area,
    #                     'bbox': self._get_bbox(mask_np)
    #                 })
            
    #         if not masks:
    #             print("[MaskGenerator] WARN: No valid segments after processing.")
    #             return None
            
    #         print(f"[MaskGenerator] FastSAM generated {len(masks)} segments. Applying '{strategy}' strategy.")
            
    #         # --- Strategy Logic ---
    #         background_ann = None
    #         if strategy == 'border':
    #             border_masks = [ann for ann in masks if self._touches_border(ann['segmentation'])]
    #             if border_masks:
    #                 background_ann = sorted(border_masks, key=lambda x: x['area'], reverse=True)[0]
    #             else:
    #                 print("[MaskGenerator] WARN: No border masks found. Falling back to largest area mask.")
            
    #         elif strategy == 'variance':
    #             processed_masks = []
    #             for ann in masks:
    #                 if ann['area'] < 1000: 
    #                     continue
    #                 masked_pixels = image_rgb_np[ann['segmentation']]
    #                 if masked_pixels.size == 0: 
    #                     continue
    #                 color_std = np.std(masked_pixels, axis=0).mean()
    #                 score = color_std / np.log(ann['area'] + 1.01)
    #                 processed_masks.append({'ann': ann, 'score': score})
                
    #             if processed_masks:
    #                 background = sorted(processed_masks, key=lambda x: x['score'])[0]
    #                 background_ann = background['ann']
    #             else:
    #                 print("[MaskGenerator] WARN: No suitable masks for variance strategy. Falling back to largest area mask.")
            
    #         elif strategy == 'largest':
    #             # Directly select the largest mask as background
    #             background_ann = sorted(masks, key=lambda x: x['area'], reverse=True)[0]
            
    #         elif strategy == 'center':
    #             # Select the mask closest to the image center
    #             h, w = image_rgb_np.shape[:2]
    #             center_y, center_x = h // 2, w // 2
                
    #             center_distances = []
    #             for ann in masks:
    #                 mask = ann['segmentation']
    #                 if np.sum(mask) > 0:
    #                     y_coords, x_coords = np.where(mask)
    #                     centroid_y = np.mean(y_coords)
    #                     centroid_x = np.mean(x_coords)
    #                     distance = np.sqrt((centroid_y - center_y)**2 + (centroid_x - center_x)**2)
    #                     center_distances.append((ann, distance))
                
    #             if center_distances:
    #                 background_ann = sorted(center_distances, key=lambda x: x[1])[0][0]
    #             else:
    #                 print("[MaskGenerator] WARN: No suitable masks for center strategy. Falling back to largest area mask.")
            
    #         else:
    #             raise ValueError(f"Unknown mask strategy: {strategy}")

    #         # Fallback if strategy failed: use the largest mask as background
    #         if background_ann is None:
    #             background_ann = sorted(masks, key=lambda x: x['area'], reverse=True)[0]

    #         # Invert the background mask to get the subject mask
    #         subject_mask = ~background_ann['segmentation']
            
    #         subject_ratio = subject_mask.sum() / subject_mask.size * 100
    #         print(f"[MaskGenerator] Mask generated. Subject area: {subject_ratio:.2f}%")
            
    #         return subject_mask
            
    #     except Exception as e:
    #         print(f"[MaskGenerator] ERROR during mask generation: {e}")
    #         return None

    def generate_mask_from_np(self, image_rgb_np, **kwargs):
        """
        Generate a subject binary mask from a NumPy RGB image array
        
        Args:
            image_rgb_np (np.ndarray): NumPy format RGB image, shape [H, W, 3], dtype uint8
            
        Returns:
            np.ndarray: boolean mask of shape [H, W], True indicates subject, None on failure
        """
        if self.mask_generator is None:
            print("[MaskGenerator] ERROR: FastSAM model not available for mask generation.")
            return None

        print(f"[MaskGenerator] Generating segments for a {image_rgb_np.shape} image using FastSAM...")
        
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb_np).convert("RGB")
            
            # FastSAM inference
            everything_results = self.fastsam(
                pil_image,
                device=self.device,
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9
            )
            
            # Create prompt processor and get all segmentation results
            prompt_process = FastSAMPrompt(pil_image, everything_results, device=self.device)
            masks_tensor = prompt_process.everything_prompt()
            
            if masks_tensor is None or masks_tensor.shape[0] == 0:
                print("[MaskGenerator] WARN: FastSAM did not produce any segments.")
                return None
            
            print(f"[MaskGenerator] FastSAM generated {masks_tensor.shape[0]} segments.")
            
            # Combine all masks using logical OR (any mask True means subject)
            combined_mask = torch.any(masks_tensor, dim=0)
            
            # Convert to numpy boolean mask
            subject_mask = combined_mask.cpu().numpy().astype(bool)
            
            # Calculate subject area ratio
            subject_ratio = subject_mask.sum() / subject_mask.size * 100
            print(f"[MaskGenerator] Mask generated. Subject area: {subject_ratio:.2f}%")
            
            return subject_mask
            
        except Exception as e:
            print(f"[MaskGenerator] ERROR during mask generation: {e}")
            return None
    
    def _touches_border(self, mask):
        """Check whether the mask touches the image border"""
        return (mask[0, :].any() or mask[-1, :].any() or 
                mask[:, 0].any() or mask[:, -1].any())
    
    def _get_bbox(self, mask):
        """Get the bounding box of the mask"""
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return [0, 0, 0, 0]
        return [np.min(cols), np.min(rows), np.max(cols), np.max(rows)]