import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image
from torch import zeros_like
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import glob
import re
from .preprocess_files import face_mask_google_mediapipe
from typing import Optional, List, Dict, Any

OBJECT_TEMPLATE = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

STYLE_TEMPLATE = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

NULL_TEMPLATE = ["{}"]

TEMPLATE_MAP = {
    "object": OBJECT_TEMPLATE,
    "style": STYLE_TEMPLATE,
    "null": NULL_TEMPLATE,
}


def _randomset(lis):
    ret = []
    for i in range(len(lis)):
        if random.random() < 0.5:
            ret.append(lis[i])
    return ret


def _shuffle(lis):

    return random.sample(lis, len(lis))


def _get_cutout_holes(
    height,
    width,
    min_holes=8,
    max_holes=32,
    min_height=16,
    max_height=128,
    min_width=16,
    max_width=128,
):
    holes = []
    for _n in range(random.randint(min_holes, max_holes)):
        hole_height = random.randint(min_height, max_height)
        hole_width = random.randint(min_width, max_width)
        y1 = random.randint(0, height - hole_height)
        x1 = random.randint(0, width - hole_width)
        y2 = y1 + hole_height
        x2 = x1 + hole_width
        holes.append((x1, y1, x2, y2))
    return holes


def _generate_random_mask(image):
    mask = zeros_like(image[:1])
    holes = _get_cutout_holes(mask.shape[1], mask.shape[2])
    for (x1, y1, x2, y2) in holes:
        mask[:, y1:y2, x1:x2] = 1.0
    if random.uniform(0, 1) < 0.25:
        mask.fill_(1.0)
    masked_image = image * (mask < 0.5)
    return mask, masked_image


class PivotalTuningDatasetCapation(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        token_map: Optional[dict] = None,
        use_template: Optional[str] = None,
        size=512,
        h_flip=True,
        color_jitter=False,
        resize=True,
        use_mask_captioned_data=False,
        use_face_segmentation_condition=False,
        train_inpainting=False,
        blur_amount: int = 70,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.resize = resize
        self.train_inpainting = train_inpainting

        instance_data_root = Path(instance_data_root)
        if not instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = []
        self.mask_path = []

        assert not (
            use_mask_captioned_data and use_template
        ), "Can't use both mask caption data and template."

        # Prepare the instance images
        if use_mask_captioned_data:
            src_imgs = glob.glob(str(instance_data_root) + "/*src.jpg")
            for f in src_imgs:
                idx = int(str(Path(f).stem).split(".")[0])
                mask_path = f"{instance_data_root}/{idx}.mask.png"

                if Path(mask_path).exists():
                    self.instance_images_path.append(f)
                    self.mask_path.append(mask_path)
                else:
                    print(f"Mask not found for {f}")

            self.captions = open(f"{instance_data_root}/caption.txt").readlines()

        else:
            possibily_src_images = (
                glob.glob(str(instance_data_root) + "/*.jpg")
                + glob.glob(str(instance_data_root) + "/*.png")
                + glob.glob(str(instance_data_root) + "/*.jpeg")
            )
            possibily_src_images = (
                set(possibily_src_images)
                - set(glob.glob(str(instance_data_root) + "/*mask.png"))
                - set([str(instance_data_root) + "/caption.txt"])
            )

            self.instance_images_path = list(set(possibily_src_images))
            self.captions = [
                x.split("/")[-1].split(".")[0] for x in self.instance_images_path
            ]

        assert (
            len(self.instance_images_path) > 0
        ), "No images found in the instance data root."

        self.instance_images_path = sorted(self.instance_images_path)

        self.use_mask = use_face_segmentation_condition or use_mask_captioned_data
        self.use_mask_captioned_data = use_mask_captioned_data

        if use_face_segmentation_condition:

            for idx in range(len(self.instance_images_path)):
                targ = f"{instance_data_root}/{idx}.mask.png"
                # see if the mask exists
                if not Path(targ).exists():
                    print(f"Mask not found for {targ}")

                    print(
                        "Warning : this will pre-process all the images in the instance data root."
                    )

                    if len(self.mask_path) > 0:
                        print(
                            "Warning : masks already exists, but will be overwritten."
                        )

                    masks = face_mask_google_mediapipe(
                        [
                            Image.open(f).convert("RGB")
                            for f in self.instance_images_path
                        ]
                    )
                    for idx, mask in enumerate(masks):
                        mask.save(f"{instance_data_root}/{idx}.mask.png")

                    break

            for idx in range(len(self.instance_images_path)):
                self.mask_path.append(f"{instance_data_root}/{idx}.mask.png")

        self.num_instance_images = len(self.instance_images_path)
        self.token_map = token_map

        self.use_template = use_template
        if use_template is not None:
            self.templates = TEMPLATE_MAP[use_template]

        self._length = self.num_instance_images

        self.h_flip = h_flip
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                )
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(0.1, 0.1)
                if color_jitter
                else transforms.Lambda(lambda x: x),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.blur_amount = blur_amount

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.train_inpainting:
            (
                example["instance_masks"],
                example["instance_masked_images"],
            ) = _generate_random_mask(example["instance_images"])

        if self.use_template:
            assert self.token_map is not None
            input_tok = list(self.token_map.values())[0]

            text = random.choice(self.templates).format(input_tok)
        else:
            text = self.captions[index % self.num_instance_images].strip()

            if self.token_map is not None:
                for token, value in self.token_map.items():
                    text = text.replace(token, value)

        print(text)

        if self.use_mask:
            example["mask"] = (
                self.image_transforms(
                    Image.open(self.mask_path[index % self.num_instance_images])
                )
                * 0.5
                + 1.0
            )

        if self.h_flip and random.random() > 0.5:
            hflip = transforms.RandomHorizontalFlip(p=1)

            example["instance_images"] = hflip(example["instance_images"])
            if self.use_mask:
                example["mask"] = hflip(example["mask"])

        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example

class PivotalTuningDatasetCapationPromptOnly(Dataset):
    """
    A dataset that returns tokenized prompts from a student tokenizer
    and two teacher tokenizers.
    """

    def __init__(
        self,
        student_tokenizer, # Student's tokenizer
        teacher_tokenizer1, # First teacher's tokenizer
        teacher_tokenizer2, # Second teacher's tokenizer
        instance_data_root: Optional[str] = None,
        token_map: Optional[Dict[str, str]] = None,
        use_template: Optional[str] = None,
        captions: Optional[List[str]] = None,
        dataset_size: int = 10,
    ):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer1 = teacher_tokenizer1
        self.teacher_tokenizer2 = teacher_tokenizer2

        self.token_map = token_map
        self.use_template = use_template
        self.dataset_size = dataset_size

        # Generate captions list
        if captions is not None:
            self.captions = [str(c).strip() for c in captions]  # Ensure strings and strip whitespace
            if not self.captions:  # If the provided captions list is empty
                 print("Warning: Provided captions list is empty. Falling back to default 'gaussian noise' prompts.")
                 self.captions = [f"gaussian noise {i}" for i in range(self.dataset_size)]
            elif len(self.captions) < self.dataset_size:
                # Ensure captions list is long enough to cover dataset_size
                factor = (self.dataset_size // len(self.captions)) + 1
                self.captions = (self.captions * factor)[:self.dataset_size]
        elif instance_data_root is not None:
            from pathlib import Path
            caption_path = Path(instance_data_root) / "caption.txt"
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:  # Specify encoding
                    self.captions = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines
                if not self.captions:
                    print(f"Warning: Caption file {caption_path} is empty or contains only whitespace. Falling back to default prompts.")
                    self.captions = [f"gaussian noise {i}" for i in range(self.dataset_size)]
                elif len(self.captions) < self.dataset_size:
                    factor = (self.dataset_size // len(self.captions)) + 1
                    self.captions = (self.captions * factor)[:self.dataset_size]
            else:
                print(f"Warning: Caption file {caption_path} not found. Falling back to default 'gaussian noise' prompts.")
                self.captions = [f"gaussian noise {i}" for i in range(self.dataset_size)]
        else:
            self.captions = [f"gaussian noise {i}" for i in range(self.dataset_size)]

        if self.use_template is not None:
            # Ensure TEMPLATE_MAP is defined
            if 'TEMPLATE_MAP' not in globals() or not isinstance(TEMPLATE_MAP, dict) or use_template not in TEMPLATE_MAP:
                raise ValueError(f"TEMPLATE_MAP is not defined or does not contain key '{use_template}'.")
            self.templates = TEMPLATE_MAP[use_template]
            if not self.templates:
                raise ValueError(f"No templates found for '{use_template}' in TEMPLATE_MAP.")

        self._length = self.dataset_size  # Length is determined by dataset_size

    def __len__(self):
        return self._length

    def _tokenize_text(self, tokenizer, text, padding_strategy="do_not_pad"):
        # Encapsulate tokenize logic for reusability and easier padding strategy modification
        # Note: Hugging Face tokenizer max_length refers to token count, not character count
        # tokenizer.model_max_length is usually a good choice
        return tokenizer(
            text,
            padding=padding_strategy,  # "max_length" or "do_not_pad"
            truncation=True,
            max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 77,  # CLIP typically uses 77
            return_tensors=None,  # Return list, convert to tensor in collate_fn
        ).input_ids

    def __getitem__(self, index):
        example = {}
        # Get the original text for the current index
        # Use index % len(self.captions) to safely cycle through captions,
        # even if self.dataset_size > len(self.captions) (handled during initialization)
        current_caption_idx = index % len(self.captions)
        base_text = self.captions[current_caption_idx]

        if self.use_template:
            if self.token_map is None:
                raise ValueError("token_map must be provided when use_template is set.")
            # Ensure token_map is not empty
            if not self.token_map:
                 raise ValueError("token_map is empty, but use_template is set.")
            input_tok = list(self.token_map.values())[0]  # Assume token_map has at least one element
            text_to_tokenize = random.choice(self.templates).format(input_tok)
            
        else:
            text_to_tokenize = base_text
            if self.token_map is not None:
                for token, value in self.token_map.items():
                    text_to_tokenize = text_to_tokenize.replace(token, value)
        
        print(text_to_tokenize)

        # Use student tokenizer
        example["student_prompt_ids"] = self._tokenize_text(self.student_tokenizer, text_to_tokenize)

        # Use teacher tokenizer 1
        example["teacher1_prompt_ids"] = self._tokenize_text(self.teacher_tokenizer1, text_to_tokenize)

        # Use teacher tokenizer 2
        example["teacher2_prompt_ids"] = self._tokenize_text(self.teacher_tokenizer2, text_to_tokenize)
        
        # Optionally return the original text for debugging
        example["raw_text"] = text_to_tokenize

        return example

class PivotalTuningDatasetCapationLoraGenerated(Dataset):
    """
    A dataset class that uses a LoRA-loaded Stable Diffusion pipeline to dynamically generate images
    and tokenizes the corresponding prompt text.
    It also supports saving generated images to disk for debugging and inspection.
    The saved images are the raw PIL Images directly output by the pipeline.
    """

    def __init__(
        self,
        # --- SD Pipeline and generation parameters ---
        sd_pipeline: Any,  # Stable Diffusion Pipeline (e.g., StableDiffusionPipeline)
        main_tokenizer: Any,  # Main tokenizer (e.g., CLIPTokenizer)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        generation_height: int = 512,
        generation_width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generation_seed: Optional[int] = None,

        # --- Prompt text tokenization parameters ---
        aux_tokenizer1: Optional[Any] = None,
        aux_tokenizer2: Optional[Any] = None,
        instance_data_root: Optional[str] = None,
        token_map: Optional[Dict[str, str]] = None,
        use_template: Optional[str] = None,
        captions: Optional[List[str]] = None,
        dataset_size: int = 100,

        # --- Image transformation parameters (applied to images fed into the model) ---
        transform_size: int = 512,
        h_flip: bool = True,
        color_jitter_strength: float = 0.0,
        resize_output: bool = True,  # Resize for the tensor fed to model, not necessarily for saved image

        # --- Image saving parameters ---
        save_generated_images_path: Optional[str] = None,  # Path to directory for saving generated images
        save_image_prefix: str = "gen_img",               # Prefix for saved image filenames
    ):
        self.sd_pipeline = sd_pipeline.to(device)
        self.device = device
        self.generation_height = generation_height
        self.generation_width = generation_width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.generation_seed = generation_seed

        self.main_tokenizer = main_tokenizer
        self.aux_tokenizer1 = aux_tokenizer1
        self.aux_tokenizer2 = aux_tokenizer2

        self.token_map = token_map
        self.use_template = use_template
        self.dataset_size = dataset_size

        # --- Caption loading logic ---
        if captions is not None:
            self.captions_list = [str(c).strip() for c in captions if str(c).strip()]
            if not self.captions_list:
                print("Warning: Provided captions list is empty or contains only whitespace. Falling back to default prompts.")
                default_token_val = list(token_map.values())[0] if token_map and token_map.values() else 'target_object'
                self.captions_list = [f"prompt for <{default_token_val}> {i}" for i in range(10)]
        elif instance_data_root is not None:
            instance_data_root_path = Path(instance_data_root)
            caption_path = instance_data_root_path / "caption.txt"
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    self.captions_list = [line.strip() for line in f.readlines() if line.strip()]
                if not self.captions_list:
                    print(f"Warning: Caption file {caption_path} is empty. Falling back to default prompts.")
                    default_token_val = list(token_map.values())[0] if token_map and token_map.values() else 'target_object'
                    self.captions_list = [f"prompt for <{default_token_val}> {i}" for i in range(10)]
            else:
                print(f"Warning: Caption file {caption_path} not found. Falling back to default prompts.")
                default_token_val = list(token_map.values())[0] if token_map and token_map.values() else 'target_object'
                self.captions_list = [f"prompt for <{default_token_val}> {i}" for i in range(10)]
        else:
            print("Warning: No captions or instance_data_root provided. Falling back to default prompts.")
            default_token_val = list(token_map.values())[0] if token_map and token_map.values() else 'target_object'
            self.captions_list = [f"prompt for <{default_token_val}> {i}" for i in range(10)]

        if self.captions_list and len(self.captions_list) < self.dataset_size:
            factor = (self.dataset_size // len(self.captions_list)) + 1
            self.captions_list = (self.captions_list * factor)
        if not self.captions_list:  # Fallback if still empty
             self.captions_list = [f"default prompt {i}" for i in range(self.dataset_size if self.dataset_size > 0 else 10)]

        if self.use_template is not None:
            if 'TEMPLATE_MAP' not in globals() or not isinstance(TEMPLATE_MAP, dict) or use_template not in TEMPLATE_MAP:
                # Fallback if TEMPLATE_MAP is not globally defined or key is missing
                print(f"Warning: TEMPLATE_MAP is not defined or does not contain key '{use_template}'. Using basic prompt.")
                self.templates = None  # Indicate no templates available
                self.use_template = None  # Disable template usage
            else:
                self.templates = TEMPLATE_MAP[use_template]
                if not self.templates:
                    print(f"Warning: No templates found for '{use_template}' in TEMPLATE_MAP. Using basic prompt.")
                    self.use_template = None  # Disable template usage

        self._length = self.dataset_size

        # --- Image transforms (applied to images fed into the model) ---
        self.h_flip = h_flip
        transform_list = []
        if resize_output: # This resize is for the tensor output
            transform_list.append(
                transforms.Resize(transform_size, interpolation=transforms.InterpolationMode.BILINEAR)
            )
        if color_jitter_strength > 0:
            transform_list.append(transforms.ColorJitter(brightness=color_jitter_strength, contrast=color_jitter_strength))
        transform_list.extend([
            transforms.CenterCrop(transform_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
        self.image_transforms = transforms.Compose(transform_list)
        
        # --- Initialize image saving settings ---
        self.save_generated_images_path = None
        if save_generated_images_path:
            self.save_generated_images_path = Path(save_generated_images_path)
            self.save_generated_images_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
            print(f"Generated images will be saved to: {self.save_generated_images_path}")
        self.save_image_prefix = save_image_prefix


    def __len__(self):
        return self._length

    def _tokenize_text(self, tokenizer, text, padding_strategy="do_not_pad"):
        if tokenizer is None:
            return None
        return tokenizer(
            text,
            padding=padding_strategy,
            truncation=True,
            max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 77,
            return_tensors=None, # Return list of token_ids
        ).input_ids
    
    # _tensor_to_pil is no longer strictly needed for the primary save operation,
    # but can be kept for other debugging/utility purposes.
    def _tensor_to_pil(self, tensor_image):
        """Convert a normalized [-1, 1] tensor back to a PIL Image (0-255 range)."""
        tensor_image = (tensor_image / 2.0) + 0.5
        pil_image = transforms.ToPILImage()(tensor_image.cpu())
        return pil_image

    def _clean_filename_prompt(self, prompt_text, max_len=50):
        """Clean prompt text to make it suitable as part of a filename."""
        cleaned = re.sub(r'[^\w\s-]', '', prompt_text).strip().replace(' ', '_')
        return cleaned[:max_len]

    def __getitem__(self, index):
        example = {}

        # --- 1. Get the base caption ---
        current_caption_idx = index % len(self.captions_list)
        base_text = self.captions_list[current_caption_idx]

        # --- 2. Build the final prompt text ---
        if self.use_template and self.templates:  # Check if templates are available
            if self.token_map is None or not self.token_map:
                # This case should ideally be an error, but for robustness, fallback
                print("Warning: Non-empty token_map must be provided when use_template is set. Falling back to base text.")
                text_to_use = base_text
            else:
                placeholder_value = list(self.token_map.values())[0]
                text_to_use = random.choice(self.templates).format(placeholder_value)
        else:
            text_to_use = base_text
            if self.token_map is not None:
                for token, value in self.token_map.items():
                    text_to_use = text_to_use.replace(token, value)
        
        # --- 3. Generate image ---
        generator = None
        if self.generation_seed is not None:
            # Ensure seed varies per item if a base seed is given
            generator = torch.Generator(device=self.device).manual_seed(42)

        with torch.no_grad():
            # This is the raw PIL Image directly from the pipeline
            torch.manual_seed(42)
            generated_image_pil_original = self.sd_pipeline(
                prompt=text_to_use,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            ).images[0]
            print(text_to_use)

        # --- 4. Save the original generated image (if a path is specified) ---
        # This step occurs before any image transforms
        if self.save_generated_images_path:
            prompt_suffix = self._clean_filename_prompt(text_to_use)
            # Build filename: prefix_index_prompt_summary.png
            # Note: the saved image is the original before any transforms (including training flips)
            filename_parts = [
                self.save_image_prefix,
                f"{index:04d}",  # Zero-padded index for sorting
            ]
            if prompt_suffix:
                filename_parts.append(prompt_suffix)

            save_filename = "_".join(filename_parts) + ".png"
            full_save_path = self.save_generated_images_path / save_filename

            try:
                # Directly save the PIL Image object
                generated_image_pil_original.save(full_save_path)
            except Exception as e:
                print(f"Error: Could not save original image to {full_save_path}: {e}")

            torch.cuda.empty_cache()

        # --- 5. Apply image transforms to get the final image tensor for training ---
        # These transforms prepare the input for the model and do not affect the saved original image
        instance_image_tensor = self.image_transforms(generated_image_pil_original)  # Tensor range [-1, 1]

        # --- 6. (Optional) Horizontal flip of training image tensor ---
        if self.h_flip and random.random() > 0.5:
            hflip_transform = transforms.RandomHorizontalFlip(p=1.0)
            instance_image_tensor = hflip_transform(instance_image_tensor)
            # Note: this flip is only applied to the tensor fed into the model;
            # the saved image `generated_image_pil_original` remains unflipped.

        example["instance_images"] = instance_image_tensor

        # --- 7. Tokenize prompt text ---
        example["instance_prompt_ids"] = self._tokenize_text(self.main_tokenizer, text_to_use)
        if self.aux_tokenizer1:
            example["aux1_prompt_ids"] = self._tokenize_text(self.aux_tokenizer1, text_to_use)
        if self.aux_tokenizer2:
            example["aux2_prompt_ids"] = self._tokenize_text(self.aux_tokenizer2, text_to_use)
        example["raw_text"] = text_to_use
        
        return example
