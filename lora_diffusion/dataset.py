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

# OBJECT_TEMPLATE = [
#     "a highly detailed metallic {} with visible surface texture",
#     "a machined metal {} with precise engineering details",
#     "a brushed steel {} with intricate mechanical components",
#     "an industrial {} with realistic metal reflections",
#     "a chrome-plated {} with fine surface details",
#     "a polished aluminum {} with visible manufacturing precision",
#     "a weathered metal {} with authentic patina and texture",
#     "an ultra-detailed {} with realistic metal grain patterns",
#     "a titanium-finished {} with complex mechanical structure",
#     "an anodized metal {} with precise technical specifications",
#     "a high-resolution render of a metallic {} with surface imperfections",
#     "a professional photograph of an industrial {} with visible welding seams",
#     "a cast iron {} with rich textural details",
#     "a stainless steel {} with realistic light reflections",
#     "a CNC-machined {} with detailed edge beveling",
#     "a satin-finish metal {} with intricate engravings",
#     "a forged metal {} with authentic hammer marks",
#     "an industrial-grade {} with detailed rivets and fasteners",
#     "a galvanized metal {} with realistic zinc coating texture",
#     "a copper-finished {} with natural oxidation details",
#     "a die-cast {} with microscopic surface details",
#     "an 8K ultra-detailed photograph of a metallic {}",
#     "a high-contrast image of a brass {} with fine engraving",
#     "a professional studio photograph of a machined {} with visible tooling marks",
#     "a close-up of a burnished metal {} with intricate detailing",
#     "an engineering photo of a metal {} showing precise fabrication quality",
#     "a powder-coated {} with uniform surface texture"
# ]

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
        dataset_size: int = 10, # 改个名字，noise_count 容易误解，或者叫 num_samples
    ):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer1 = teacher_tokenizer1
        self.teacher_tokenizer2 = teacher_tokenizer2

        self.token_map = token_map
        self.use_template = use_template
        self.dataset_size = dataset_size

        # 生成 captions 列表
        if captions is not None:
            self.captions = [str(c).strip() for c in captions] #确保是字符串且去除首尾空格
            if not self.captions: # 如果传入的captions是空列表
                 print("Warning: Provided captions list is empty. Falling back to default 'gaussian noise' prompts.")
                 self.captions = [f"gaussian noise {i}" for i in range(self.dataset_size)]
            elif len(self.captions) < self.dataset_size:
                # 确保 captions 列表足够长以覆盖 dataset_size
                factor = (self.dataset_size // len(self.captions)) + 1
                self.captions = (self.captions * factor)[:self.dataset_size]
        elif instance_data_root is not None:
            from pathlib import Path
            caption_path = Path(instance_data_root) / "caption.txt"
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f: # 指定编码
                    self.captions = [line.strip() for line in f.readlines() if line.strip()] # 去除空行
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
            # 确保 TEMPLATE_MAP 已定义
            if 'TEMPLATE_MAP' not in globals() or not isinstance(TEMPLATE_MAP, dict) or use_template not in TEMPLATE_MAP:
                raise ValueError(f"TEMPLATE_MAP is not defined or does not contain key '{use_template}'.")
            self.templates = TEMPLATE_MAP[use_template]
            if not self.templates:
                raise ValueError(f"No templates found for '{use_template}' in TEMPLATE_MAP.")

        self._length = self.dataset_size # 长度由 dataset_size 决定

    def __len__(self):
        return self._length

    def _tokenize_text(self, tokenizer, text, padding_strategy="do_not_pad"):
        # 封装tokenize逻辑，方便复用和修改padding策略
        # 注意：Hugging Face tokenizer的 max_length 通常是指 token 的数量，不是字符数
        # tokenizer.model_max_length 可能是个好选择
        return tokenizer(
            text,
            padding=padding_strategy, # "max_length" 或 "do_not_pad"
            truncation=True,
            max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 77, # CLIP通常是77
            return_tensors=None, # 先不转成tensor，在collate_fn里处理
        ).input_ids

    def __getitem__(self, index):
        example = {}
        # 获取当前索引对应的原始文本
        # 使用 index % len(self.captions) 来安全地循环访问 captions，
        # 即使 self.dataset_size > len(self.captions) (理论上初始化时已处理)
        current_caption_idx = index % len(self.captions)
        base_text = self.captions[current_caption_idx]

        if self.use_template:
            if self.token_map is None:
                raise ValueError("token_map must be provided when use_template is set.")
            # 确保 token_map 非空
            if not self.token_map:
                 raise ValueError("token_map is empty, but use_template is set.")
            input_tok = list(self.token_map.values())[0] # 假设 token_map 至少有一个元素
            text_to_tokenize = random.choice(self.templates).format(input_tok)
            
        else:
            text_to_tokenize = base_text
            if self.token_map is not None:
                for token, value in self.token_map.items():
                    text_to_tokenize = text_to_tokenize.replace(token, value)
        
        print(text_to_tokenize)

        # 使用 student tokenizer
        example["student_prompt_ids"] = self._tokenize_text(self.student_tokenizer, text_to_tokenize)

        # 使用 teacher tokenizer 1
        example["teacher1_prompt_ids"] = self._tokenize_text(self.teacher_tokenizer1, text_to_tokenize)

        # 使用 teacher tokenizer 2
        example["teacher2_prompt_ids"] = self._tokenize_text(self.teacher_tokenizer2, text_to_tokenize)
        
        # 你可能还想返回原始文本，方便调试
        example["raw_text"] = text_to_tokenize

        return example

class PivotalTuningDatasetCapationLoraGenerated(Dataset):
    """
    一个数据集类，它使用一个加载了 LoRA 的 Stable Diffusion pipeline 来动态生成图像，
    并对相应的提示文本进行分词。
    它还增加了将生成的图像保存到磁盘的功能，方便调试和查看。
    保存的图像是 pipeline 直接输出的原始 PIL Image。
    """

    def __init__(
        self,
        # --- SD Pipeline 和生成参数 ---
        sd_pipeline: Any, # Stable Diffusion Pipeline (e.g., StableDiffusionPipeline)
        main_tokenizer: Any, # Main tokenizer (e.g., CLIPTokenizer)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        generation_height: int = 512,
        generation_width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generation_seed: Optional[int] = None,

        # --- 提示文本分词参数 ---
        aux_tokenizer1: Optional[Any] = None,
        aux_tokenizer2: Optional[Any] = None,
        instance_data_root: Optional[str] = None,
        token_map: Optional[Dict[str, str]] = None,
        use_template: Optional[str] = None,
        captions: Optional[List[str]] = None,
        dataset_size: int = 100,

        # --- 图像变换参数 (应用于送入模型的图像) ---
        transform_size: int = 512,
        h_flip: bool = True,
        color_jitter_strength: float = 0.0,
        resize_output: bool = True, # Resize for the tensor fed to model, not necessarily for saved image

        # --- 新增：图像保存参数 ---
        save_generated_images_path: Optional[str] = None, # 指定保存生成图像的目录路径
        save_image_prefix: str = "gen_img",              # 保存图像文件名的前缀
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

        # --- 标题加载逻辑 ---
        if captions is not None:
            self.captions_list = [str(c).strip() for c in captions if str(c).strip()]
            if not self.captions_list:
                print("警告: 提供的 captions 列表为空或仅包含空白。将回退到默认提示。")
                default_token_val = list(token_map.values())[0] if token_map and token_map.values() else '目标物体'
                self.captions_list = [f"关于 <{default_token_val}> 的提示 {i}" for i in range(10)]
        elif instance_data_root is not None:
            instance_data_root_path = Path(instance_data_root)
            caption_path = instance_data_root_path / "caption.txt"
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    self.captions_list = [line.strip() for line in f.readlines() if line.strip()]
                if not self.captions_list:
                    print(f"警告: 标题文件 {caption_path} 为空。将回退到默认提示。")
                    default_token_val = list(token_map.values())[0] if token_map and token_map.values() else '目标物体'
                    self.captions_list = [f"关于 <{default_token_val}> 的提示 {i}" for i in range(10)]
            else:
                print(f"警告: 标题文件 {caption_path} 未找到。将回退到默认提示。")
                default_token_val = list(token_map.values())[0] if token_map and token_map.values() else '目标物体'
                self.captions_list = [f"关于 <{default_token_val}> 的提示 {i}" for i in range(10)]
        else:
            print("警告: 未提供 captions 或 instance_data_root。将回退到默认提示。")
            default_token_val = list(token_map.values())[0] if token_map and token_map.values() else '目标物体'
            self.captions_list = [f"关于 <{default_token_val}> 的提示 {i}" for i in range(10)]

        if self.captions_list and len(self.captions_list) < self.dataset_size:
            factor = (self.dataset_size // len(self.captions_list)) + 1
            self.captions_list = (self.captions_list * factor)
        if not self.captions_list: # Fallback if still empty
             self.captions_list = [f"默认提示 {i}" for i in range(self.dataset_size if self.dataset_size > 0 else 10)]

        if self.use_template is not None:
            if 'TEMPLATE_MAP' not in globals() or not isinstance(TEMPLATE_MAP, dict) or use_template not in TEMPLATE_MAP:
                # Fallback if TEMPLATE_MAP is not globally defined or key is missing
                print(f"警告: TEMPLATE_MAP 未定义或不包含键 '{use_template}'. 将使用基本提示.")
                self.templates = None # Indicate no templates available
                self.use_template = None # Disable template usage
            else:
                self.templates = TEMPLATE_MAP[use_template]
                if not self.templates:
                    print(f"警告: 在 TEMPLATE_MAP 中未找到 '{use_template}' 的模板。将使用基本提示.")
                    self.use_template = None # Disable template usage

        self._length = self.dataset_size

        # --- 图像变换 (应用于送入模型的图像) ---
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
            transforms.Normalize([0.5], [0.5]), # 归一化到 [-1, 1]
        ])
        self.image_transforms = transforms.Compose(transform_list)
        
        # --- 初始化图像保存设置 ---
        self.save_generated_images_path = None
        if save_generated_images_path:
            self.save_generated_images_path = Path(save_generated_images_path)
            self.save_generated_images_path.mkdir(parents=True, exist_ok=True) # 创建目录（如果不存在）
            print(f"生成的图像将被保存在: {self.save_generated_images_path}")
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
        """将归一化的 [-1, 1] 张量转换回 PIL Image (0-255范围)"""
        tensor_image = (tensor_image / 2.0) + 0.5
        pil_image = transforms.ToPILImage()(tensor_image.cpu())
        return pil_image

    def _clean_filename_prompt(self, prompt_text, max_len=50):
        """清理提示文本，使其适合作为文件名的一部分"""
        cleaned = re.sub(r'[^\w\s-]', '', prompt_text).strip().replace(' ', '_')
        return cleaned[:max_len]

    def __getitem__(self, index):
        example = {}

        # --- 1. 获取基础标题 ---
        current_caption_idx = index % len(self.captions_list)
        base_text = self.captions_list[current_caption_idx]

        # --- 2. 构建最终的提示文本 ---
        if self.use_template and self.templates: # Check if templates are available
            if self.token_map is None or not self.token_map:
                # This case should ideally be an error, but for robustness, fallback
                print("警告: 当 use_template 设置时，必须提供非空的 token_map。将使用基础文本。")
                text_to_use = base_text
            else:
                placeholder_value = list(self.token_map.values())[0]
                text_to_use = random.choice(self.templates).format(placeholder_value)
        else:
            text_to_use = base_text
            if self.token_map is not None:
                for token, value in self.token_map.items():
                    text_to_use = text_to_use.replace(token, value)
        
        # --- 3. 生成图像 ---
        generator = None
        if self.generation_seed is not None:
            # Ensure seed varies per item if a base seed is given
            generator = torch.Generator(device=self.device).manual_seed(42) 

        # print(f"Before SD pipeline call (idx {index}): Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB, Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
        
        with torch.no_grad():
            # This is the raw PIL Image directly from the pipeline
            torch.manual_seed(42)
            generated_image_pil_original = self.sd_pipeline(
                prompt=text_to_use,
                # height=self.generation_height,
                # width=self.generation_width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                # generator=generator,
            ).images[0]
            print(text_to_use)
        
        # print(f"After SD pipeline call (idx {index}): Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB, Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")            

        # --- 4. 保存原始生成的图像 (如果指定了路径) ---
        #    此步骤现在发生在任何图像变换之前
        if self.save_generated_images_path:
            prompt_suffix = self._clean_filename_prompt(text_to_use)
            # 构建文件名：前缀_索引_提示摘要.png
            # 注意: 这里保存的是原始生成的图像，在任何变换（包括模型训练用的翻转）之前
            filename_parts = [
                self.save_image_prefix,
                f"{index:04d}", # 使用0填充的索引，方便排序
            ]
            if prompt_suffix:
                filename_parts.append(prompt_suffix)
            
            save_filename = "_".join(filename_parts) + ".png"
            full_save_path = self.save_generated_images_path / save_filename
            
            try:
                # 直接保存 PIL Image 对象
                generated_image_pil_original.save(full_save_path)
                # print(f"已保存原始图像: {full_save_path}") # 如果需要可以取消注释
            except Exception as e:
                print(f"错误：无法保存原始图像到 {full_save_path}: {e}")
                
            torch.cuda.empty_cache()
            # 可以再打印一次显存，看看清空缓存后的效果
            # print(f"After empty_cache (idx {index}): Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB, Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

        # --- 5. 应用图像变换，得到用于训练的最终图像张量 ---
        #    这些变换是为模型输入准备的，不影响上面已保存的原始图像
        instance_image_tensor = self.image_transforms(generated_image_pil_original) # 张量范围 [-1, 1]

        # --- 6. (可选) 对训练图像张量进行水平翻转 ---
        if self.h_flip and random.random() > 0.5:
            hflip_transform = transforms.RandomHorizontalFlip(p=1.0)
            instance_image_tensor = hflip_transform(instance_image_tensor)
            # 注意: 这个翻转只应用于送入模型的张量，
            # 已保存的图像 `generated_image_pil_original` 是未翻转的。
            # 如果希望文件名反映这一点，需要更复杂的逻辑或保存变换后的图像。

        example["instance_images"] = instance_image_tensor

        # --- 7. 对提示文本进行分词 ---
        example["instance_prompt_ids"] = self._tokenize_text(self.main_tokenizer, text_to_use)
        if self.aux_tokenizer1:
            example["aux1_prompt_ids"] = self._tokenize_text(self.aux_tokenizer1, text_to_use)
        if self.aux_tokenizer2:
            example["aux2_prompt_ids"] = self._tokenize_text(self.aux_tokenizer2, text_to_use)
        example["raw_text"] = text_to_use
        
        return example
