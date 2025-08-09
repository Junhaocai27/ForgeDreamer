# mask_utils.py

import numpy as np
import torch
import cv2
import sys
import math

try:
    # 假设 'segment_anything' 在父目录或已安装
    sys.path.append("..") 
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("FATAL: 'segment_anything' library not found. Please install it or check sys.path.")
    sys.exit(1)

class SubjectMaskGenerator:
    def __init__(self, device, sam_checkpoint_path, sam_model_type='vit_b'):
        self.device = device
        self.sam = None
        self.mask_generator = None

        print("[MaskGenerator] Initializing...")
        try:
            self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
            self.sam.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(model=self.sam)
            print("[MaskGenerator] SAM model loaded successfully.")
        except Exception as e:
            print(f"[MaskGenerator] FATAL: Failed to load SAM model from {sam_checkpoint_path}. Error: {e}")
            # The calling code should handle the case where self.mask_generator is None

    def generate_mask_from_np(self, image_rgb_np, strategy='border'):
        """
        Generates a subject binary mask from a NumPy RGB image array.

        Args:
            image_rgb_np (np.ndarray): An RGB image in NumPy format, shape [H, W, 3], dtype uint8.
            strategy (str): 'border' or 'variance' for background identification.

        Returns:
            np.ndarray: A boolean mask of shape [H, W] where True is the subject, or None if failed.
        """
        if self.mask_generator is None:
            print("[MaskGenerator] ERROR: SAM model not available for mask generation.")
            return None

        print(f"[MaskGenerator] Generating segments for a {image_rgb_np.shape} image...")
        masks = self.mask_generator.generate(image_rgb_np)
        
        if not masks:
            print("[MaskGenerator] WARN: SAM did not produce any segments.")
            return None

        print(f"[MaskGenerator] SAM generated {len(masks)} segments. Applying '{strategy}' strategy.")
        
        # --- Strategy Logic ---
        background_ann = None
        if strategy == 'border':
            border_masks = [ann for ann in masks if (ann['segmentation'][0, :].any() or ann['segmentation'][-1, :].any() or ann['segmentation'][:, 0].any() or ann['segmentation'][:, -1].any())]
            if border_masks:
                background_ann = sorted(border_masks, key=lambda x: x['area'], reverse=True)[0]
            else:
                print("[MaskGenerator] WARN: No border masks found. Falling back to largest area mask.")
        
        elif strategy == 'variance':
            processed_masks = []
            for ann in masks:
                if ann['area'] < 1000: continue
                masked_pixels = image_rgb_np[ann['segmentation']]
                if masked_pixels.size == 0: continue
                color_std = np.std(masked_pixels, axis=0).mean()
                score = color_std / math.log(ann['area'] + 1.01)
                processed_masks.append({'ann': ann, 'score': score})
            
            if processed_masks:
                background = sorted(processed_masks, key=lambda x: x['score'])[0]
                background_ann = background['ann']
            else:
                print("[MaskGenerator] WARN: No suitable masks for variance strategy. Falling back to largest area mask.")
        
        else:
            raise ValueError(f"Unknown mask strategy: {strategy}")

        # Fallback if strategy failed: use the largest mask as background
        if background_ann is None:
            background_ann = sorted(masks, key=lambda x: x['area'], reverse=True)[0]

        # Invert the background mask to get the subject mask
        subject_mask = ~background_ann['segmentation']
        
        subject_ratio = subject_mask.sum() / subject_mask.size * 100
        print(f"[MaskGenerator] Mask generated. Subject area: {subject_ratio:.2f}%")
        
        return subject_mask