import argparse
import hashlib
import inspect
import itertools
import math
import os
import random
import re
from pathlib import Path
from typing import Optional, List, Literal

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import wandb
import fire
import sys

sys.path.append('/home/s414e2/CJH/Text-to-3D/LucidDreamer/lora')
from lora_diffusion import (
    PivotalTuningDatasetCapation,
    PivotalTuningDatasetCapationPromptOnly,
    extract_lora_ups_down,
    inject_trainable_lora,
    inject_trainable_lora_extended,
    inspect_lora,
    save_lora_weight,
    save_all,
    prepare_clip_model_sets,
    evaluate_pipe,
    UNET_EXTENDED_TARGET_REPLACE,
    tune_lora_scale,
    patch_pipe,
)



def get_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    revision,
    placeholder_tokens1: List[str],
    placeholder_tokens2: List[str],
    initializer_tokens1: List[str],
    initializer_tokens2: List[str],
    device="cuda:0",
):

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )

    placeholder_token_ids1 = []
    placeholder_token_ids2 = []

    for token, init_tok in zip(placeholder_tokens1, initializer_tokens1):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_ids1.append(placeholder_token_id)


        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        if init_tok.startswith("<rand"):

            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
            )
            print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

        elif init_tok == "<zero>":
            token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)

            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    for token, init_tok in zip(placeholder_tokens2, initializer_tokens2):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_ids2.append(placeholder_token_id)


        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        if init_tok.startswith("<rand"):

            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
            )
            print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

        elif init_tok == "<zero>":
            token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)

            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_name_or_path or pretrained_model_name_or_path,
        subfolder=None if pretrained_vae_name_or_path else "vae",
        revision=None if pretrained_vae_name_or_path else revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )

    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        placeholder_token_ids1,
        placeholder_token_ids2,
    )


@torch.no_grad()
def text2img_dataloader(
    train_dataset,
    train_batch_size,
    tokenizer,
    vae,
    text_encoder,
    cached_latents: bool = False,
):

    if cached_latents:
        cached_latents_dataset = []
        for idx in tqdm(range(len(train_dataset))):
            batch = train_dataset[idx]

            latents = vae.encode(
                batch["instance_images"].unsqueeze(0).to(dtype=vae.dtype).to(vae.device)
            ).latent_dist.sample()
            latents = latents * 0.18215
            batch["instance_images"] = latents.squeeze(0)
            cached_latents_dataset.append(batch)

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        if examples[0].get("mask", None) is not None:
            batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch

    if cached_latents:

        train_dataloader = torch.utils.data.DataLoader(
            cached_latents_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        print("PTI : Using cached latent.")

    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    return train_dataloader

def text2img_dataloader_noise(
    train_dataset,
    train_batch_size,
    student_tokenizer,
    teacher_tokenizer1,
    teacher_tokenizer2,
    cached_latents: bool = False,
    latent_shape=(4, 64, 64),

):


    def collate_fn(examples):
        batch = {}


        student_ids_list = [example["student_prompt_ids"] for example in examples]
        student_padded = student_tokenizer.pad(
            {"input_ids": student_ids_list},
            padding="max_length",
            max_length=student_tokenizer.model_max_length,
            return_tensors="pt",
        )
        batch["student_input_ids"] = student_padded.input_ids
        if "attention_mask" in student_padded:
            batch["student_attention_mask"] = student_padded.attention_mask


        teacher1_ids_list = [example["teacher1_prompt_ids"] for example in examples]
        teacher1_padded = teacher_tokenizer1.pad(
            {"input_ids": teacher1_ids_list},
            padding="max_length",
            max_length=teacher_tokenizer1.model_max_length,
            return_tensors="pt",
        )
        batch["teacher1_input_ids"] = teacher1_padded.input_ids
        if "attention_mask" in teacher1_padded:
            batch["teacher1_attention_mask"] = teacher1_padded.attention_mask


        teacher2_ids_list = [example["teacher2_prompt_ids"] for example in examples]
        teacher2_padded = teacher_tokenizer2.pad(
            {"input_ids": teacher2_ids_list},
            padding="max_length",
            max_length=teacher_tokenizer2.model_max_length,
            return_tensors="pt",
        )
        batch["teacher2_input_ids"] = teacher2_padded.input_ids
        if "attention_mask" in teacher2_padded:
            batch["teacher2_attention_mask"] = teacher2_padded.attention_mask


        pixel_values = [torch.randn(latent_shape) * 0.18215 for _ in examples]
        batch["pixel_values"] = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()


        if "mask" in examples[0] and examples[0]["mask"] is not None:

            try:
                batch["mask"] = torch.stack([example["mask"] for example in examples])
            except Exception as e:
                print(f"Warning: Could not stack masks. Error: {e}")


        if "raw_text" in examples[0]:
            batch["raw_text"] = [example["raw_text"] for example in examples]

        return batch


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print("Using Gaussian noise latents as instance_images.")
    return train_dataloader


def inpainting_dataloader(
    train_dataset, train_batch_size, tokenizer, vae, text_encoder
):
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        mask_values = [example["instance_masks"] for example in examples]
        masked_image_values = [
            example["instance_masked_images"] for example in examples
        ]


        if examples[0].get("class_prompt_ids", None) is not None:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            mask_values += [example["class_masks"] for example in examples]
            masked_image_values += [
                example["class_masked_images"] for example in examples
            ]

        pixel_values = (
            torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
        )
        mask_values = (
            torch.stack(mask_values).to(memory_format=torch.contiguous_format).float()
        )
        masked_image_values = (
            torch.stack(masked_image_values)
            .to(memory_format=torch.contiguous_format)
            .float()
        )

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "mask_values": mask_values,
            "masked_image_values": masked_image_values,
        }

        if examples[0].get("mask", None) is not None:
            batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_dataloader

def loss_step_gaussian_noise(
    batch,
    teacher_unet,
    student_unet,
    student_text_encoder,
    teacher_text_encoder,
    scheduler,
    t_mutliplier=1.0,
    mixed_precision=False,

    mask_temperature=1.0,
):
    weight_dtype = torch.float32

    latents = batch["pixel_values"].to(student_unet.device)
    if mixed_precision:
        latents = latents.to(dtype=torch.float16)
    else:
        latents = latents.to(dtype=torch.float32)

    bsz = latents.shape[0]

    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,),
        device=latents.device,
    )
    timesteps = timesteps.long()


    student_input_ids = batch["student_input_ids"].to(student_text_encoder.device)



    student_attention_mask = batch.get("student_attention_mask")
    if student_attention_mask is not None:
        student_attention_mask = student_attention_mask.to(student_text_encoder.device)

    if mixed_precision:
        with torch.cuda.amp.autocast(enabled=True):
            student_encoder_hidden_states = student_text_encoder(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask,
            )[0]
    else:
        student_encoder_hidden_states = student_text_encoder(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
        )[0]



    teacher_input_ids = batch["teacher1_input_ids"].to(teacher_text_encoder.device)

    teacher_attention_mask = batch.get("teacher1_attention_mask")
    if teacher_attention_mask is not None:
        teacher_attention_mask = teacher_attention_mask.to(teacher_text_encoder.device)

    with torch.no_grad():
        if mixed_precision:

            with torch.cuda.amp.autocast(enabled=True):
                teacher_encoder_hidden_states_no_grad = teacher_text_encoder(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                )[0]

        else:
            teacher_encoder_hidden_states_no_grad = teacher_text_encoder(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
            )[0]



        if mixed_precision and hasattr(teacher_unet, 'dtype') and teacher_unet.dtype == torch.float16:
             teacher_encoder_hidden_states_no_grad = teacher_encoder_hidden_states_no_grad.to(torch.float16)
        else:
             teacher_encoder_hidden_states_no_grad = teacher_encoder_hidden_states_no_grad.to(torch.float32)



        teacher_latents = latents.to(teacher_unet.device, dtype=teacher_unet.dtype if hasattr(teacher_unet, 'dtype') else weight_dtype)

        if mixed_precision:
            with torch.cuda.amp.autocast(enabled=True):
                teacher_pred = teacher_unet(
                    teacher_latents, timesteps, teacher_encoder_hidden_states_no_grad
                ).sample
        else:
            teacher_pred = teacher_unet(
                teacher_latents, timesteps, teacher_encoder_hidden_states_no_grad
            ).sample



    if mixed_precision and hasattr(student_unet, 'dtype') and student_unet.dtype == torch.float16:
        student_encoder_hidden_states_for_unet = student_encoder_hidden_states.to(torch.float16)
    else:
        student_encoder_hidden_states_for_unet = student_encoder_hidden_states.to(torch.float32)


    student_latents = latents.to(student_unet.device, dtype=student_unet.dtype if hasattr(student_unet, 'dtype') else weight_dtype)


    if mixed_precision:
        with torch.cuda.amp.autocast(enabled=True):
            student_pred = student_unet(
                student_latents, timesteps, student_encoder_hidden_states_for_unet
            ).sample
    else:
        student_pred = student_unet(
            student_latents, timesteps, student_encoder_hidden_states_for_unet
        ).sample


    if batch.get("mask", None) is not None:
        mask = batch["mask"].to(student_pred.device)

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4 or mask.shape[1] != 1:

            raise ValueError(f"Mask has unexpected shape: {mask.shape}. Expected (bsz, 1, H, W) or (bsz, H, W).")

        mask = (mask + 0.01).pow(mask_temperature)
        mask = mask / mask.max()


        if mixed_precision:
            mask = mask.to(dtype=torch.float16)
        else:
            mask = mask.to(dtype=torch.float32)

        student_pred = student_pred * mask
        teacher_pred = teacher_pred * mask


    loss = F.mse_loss(student_pred.float(), teacher_pred.float(), reduction="none")
    loss = loss.mean([1, 2, 3])
    loss = loss.mean()

    return loss

def perform_tuning(
    teacher1_unet,
    teacher2_unet,
    teacher1_text_encoder,
    teacher2_text_encoder,
    student_unet,
    vae,
    student_text_encoder,
    dataloader1,
    dataloader2,
    num_steps,
    scheduler,
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path,
    lr_scheduler_lora,
    lora_unet_target_modules,
    lora_clip_target_modules,
    mask_temperature,
    out_name: str,
    tokenizer,

    cached_latents: bool,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    class_token: str = "person",
    train_inpainting: bool = False,
):


    trainable_params = {
        "unet": sum(p.requires_grad for p in student_unet.parameters()),
        "text_encoder": sum(p.requires_grad for p in student_text_encoder.parameters())
    }
    print(f"Trainable parameters: {trainable_params}")

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    weight_dtype = torch.float16

    student_unet.train()
    student_text_encoder.train()

    if log_wandb:
        preped_clip = prepare_clip_model_sets()

    loss_sum = 0.0

    for epoch in range(math.ceil(num_steps / len(dataloader1))):
        for batch1, batch2 in zip(dataloader1, dataloader2):
            lr_scheduler_lora.step()

            optimizer.zero_grad()

            loss1 = loss_step_gaussian_noise(
                batch=batch1,
                teacher_unet=teacher1_unet,
                student_unet=student_unet,
                teacher_text_encoder=teacher1_text_encoder,
                student_text_encoder=student_text_encoder,
                scheduler=scheduler,
                t_mutliplier=0.8,
                mixed_precision=True,
                mask_temperature=mask_temperature,
            )

            loss2 = loss_step_gaussian_noise(
                batch=batch2,
                teacher_unet=teacher2_unet,
                student_unet=student_unet,
                teacher_text_encoder=teacher2_text_encoder,
                student_text_encoder=student_text_encoder,
                scheduler=scheduler,
                t_mutliplier=0.8,
                mixed_precision=True,
                mask_temperature=mask_temperature,
            )

            loss_sum = loss_sum + 0.5 * loss1.detach().item() + 0.5 * loss2.detach().item()

            loss = loss1 + loss2

            loss.backward()


            if global_step % 10 == 0:
                unet_grads = [param.grad.norm().item() for name, param in student_unet.named_parameters()
                             if param.requires_grad and param.grad is not None]
                text_grads = [param.grad.norm().item() for name, param in student_text_encoder.named_parameters()
                             if param.requires_grad and param.grad is not None]

                if len(unet_grads) > 0:
                    print(f"UNet gradient norm (avg): {sum(unet_grads)/len(unet_grads):.6f}")
                else:
                    print("WARNING: No gradients detected in UNet!")

                if len(text_grads) > 0:
                    print(f"Text encoder gradient norm (avg): {sum(text_grads)/len(text_grads):.6f}")
                else:
                    print("WARNING: No gradients detected in text_encoder!")

            torch.nn.utils.clip_grad_norm_(
                itertools.chain(student_unet.parameters(), student_text_encoder.parameters()), 1.0
            )
            optimizer.step()
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler_lora.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            global_step += 1

            if global_step % save_steps == 0:
                save_all(
                    unet=student_unet,
                    text_encoder=student_text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=os.path.join(
                        save_path, f"step_{global_step}.safetensors"
                    ),
                    target_replace_module_text=lora_clip_target_modules,
                    target_replace_module_unet=lora_unet_target_modules,
                )
                moved = (
                    torch.tensor(list(itertools.chain(*inspect_lora(student_unet).values())))
                    .mean()
                    .item()
                )

                print("LORA Unet Moved", moved)
                moved = (
                    torch.tensor(
                        list(itertools.chain(*inspect_lora(student_text_encoder).values()))
                    )
                    .mean()
                    .item()
                )

                print("LORA CLIP Moved", moved)



            if global_step >= num_steps:
                break

    save_all(
        student_unet,
        student_text_encoder,
        placeholder_token_ids=placeholder_token_ids,
        placeholder_tokens=placeholder_tokens,
        save_path=os.path.join(save_path, f"{out_name}.safetensors"),
        target_replace_module_text=lora_clip_target_modules,
        target_replace_module_unet=lora_unet_target_modules,
    )


def train(
    lora_path1: str,
    lora_path2: str,
    instance_data_dir: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
    train_text_encoder: bool = True,
    pretrained_vae_name_or_path: str = None,
    revision: Optional[str] = None,
    perform_inversion: bool = False,
    use_template: Literal[None, "object", "style"] = None,
    train_inpainting: bool = False,
    placeholder_tokens: str = "",
    placeholder_tokens1: str = "",
    placeholder_tokens2: str = "",
    placeholder_token_at_data: Optional[str] = None,
    initializer_tokens: Optional[str] = None,
    seed: int = 42,
    resolution: int = 512,
    color_jitter: bool = True,
    train_batch_size: int = 1,
    sample_batch_size: int = 1,
    max_train_steps_tuning: int = 1000,
    max_train_steps_ti: int = 1000,
    save_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = False,
    lora_rank: int = 4,
    lora_unet_target_modules={"CrossAttention", "Attention", "GEGLU"},
    lora_clip_target_modules={"CLIPSdpaAttention"},
    lora_dropout_p: float = 0.0,
    lora_scale: float = 1.0,
    use_extended_lora: bool = False,
    clip_ti_decay: bool = True,
    learning_rate_unet: float = 1e-4,
    learning_rate_text: float = 1e-5,
    learning_rate_ti: float = 5e-4,
    continue_inversion: bool = False,
    continue_inversion_lr: Optional[float] = None,
    use_face_segmentation_condition: bool = False,
    cached_latents: bool = True,
    use_mask_captioned_data: bool = False,
    mask_temperature: float = 1.0,
    scale_lr: bool = False,
    lr_scheduler: str = "linear",
    lr_warmup_steps: int = 0,
    lr_scheduler_lora: str = "linear",
    lr_warmup_steps_lora: int = 0,
    weight_decay_ti: float = 0.00,
    weight_decay_lora: float = 0.001,
    use_8bit_adam: bool = False,
    device="cuda:0",
    extra_args: Optional[dict] = None,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    wandb_project_name: str = "new_pti_project",
    wandb_entity: str = "new_pti_entity",
    proxy_token: str = "person",
    enable_xformers_memory_efficient_attention: bool = False,
    out_name: str = "final_lora",
):
    torch.manual_seed(seed)

    if log_wandb:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=f"steps_{max_train_steps_ti}_lr_{learning_rate_ti}_{instance_data_dir.split('/')[-1]}",
            reinit=True,
            config={
                **(extra_args if extra_args is not None else {}),
            },
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if len(placeholder_tokens1) == 0:
        placeholder_tokens1 = []
        print("PTI : Placeholder Tokens not given, using null token")
    elif len(placeholder_tokens2) == 0:
        placeholder_tokens2 = []
        print("PTI : Placeholder Tokens not given, using null token")
    else:
        placeholder_tokens1 = placeholder_tokens1.split("|")
        placeholder_tokens2 = placeholder_tokens2.split("|")

        assert (
            sorted(placeholder_tokens1) == placeholder_tokens1
        ), f"Placeholder tokens should be sorted. Use something like {'|'.join(sorted(placeholder_tokens))}'"

        assert (
            sorted(placeholder_tokens2) == placeholder_tokens2
        ), f"Placeholder tokens should be sorted. Use something like {'|'.join(sorted(placeholder_tokens))}'"

    if initializer_tokens is None:
        print("PTI : Initializer Tokens not given, doing random inits")
        initializer_tokens1 = ["<rand-0.017>"] * len(placeholder_tokens1)
        initializer_tokens2 = ["<rand-0.017>"] * len(placeholder_tokens2)
    else:
        initializer_tokens = initializer_tokens.split("|")

    assert len(initializer_tokens1) == len(
        placeholder_tokens1
    ), "Unequal Initializer token for Placeholder tokens."

    assert len(initializer_tokens2) == len(
        placeholder_tokens2
    ), "Unequal Initializer token for Placeholder tokens."

    if proxy_token is not None:
        class_token = proxy_token
    class_token = "".join(initializer_tokens1)

    if placeholder_token_at_data is not None:
        tok, pat = placeholder_token_at_data.split("|")
        token_map = {tok: pat}

    else:
        token_map1 = {"DUMMY": "".join(placeholder_tokens1)}
        token_map2 = {"DUMMY": "".join(placeholder_tokens2)}

    print("PTI : Placeholder Tokens", placeholder_tokens1)
    print("PTI : Placeholder Tokens", placeholder_tokens2)
    print("PTI : Initializer Tokens", initializer_tokens1)
    print("PTI : Initializer Tokens", initializer_tokens2)


    student_text_encoder, student_vae, student_unet, student_tokenizer, placeholder_token_ids1, placeholder_token_ids2 = get_models(
        pretrained_model_name_or_path,
        pretrained_vae_name_or_path,
        revision,
        placeholder_tokens1,
        placeholder_tokens2,
        initializer_tokens1,
        initializer_tokens2,
        device=device,
    )



    model_id = "/home/s414e2/CJH/Text-to-3D/LucidDreamer/stable-diffusion-2-1-base"
    teacher1_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    teacher2_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    patch_pipe(
        teacher1_pipe,
        lora_path1,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )

    tune_lora_scale(teacher1_pipe.unet, 1.0)
    tune_lora_scale(teacher1_pipe.text_encoder, 1.0)

    patch_pipe(
        teacher2_pipe,
        lora_path2,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )

    tune_lora_scale(teacher2_pipe.unet, 1.0)
    tune_lora_scale(teacher2_pipe.text_encoder, 1.0)

    teacher1_text_encoder = teacher1_pipe.text_encoder
    teacher1_vae = teacher1_pipe.vae
    teacher1_unet = teacher1_pipe.unet
    teacher1_tokenizer = teacher1_pipe.tokenizer


    teacher2_text_encoder = teacher2_pipe.text_encoder
    teacher2_vae = teacher2_pipe.vae
    teacher2_unet = teacher2_pipe.unet
    teacher2_tokenizer = teacher2_pipe.tokenizer


    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    if gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()

    if enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available

        if is_xformers_available():
            student_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if scale_lr:
        unet_lr = learning_rate_unet * gradient_accumulation_steps * train_batch_size
        text_encoder_lr = (
            learning_rate_text * gradient_accumulation_steps * train_batch_size
        )
        ti_lr = learning_rate_ti * gradient_accumulation_steps * train_batch_size
    else:
        unet_lr = learning_rate_unet
        text_encoder_lr = learning_rate_text
        ti_lr = learning_rate_ti

    train_dataset1 = PivotalTuningDatasetCapationPromptOnly(
        token_map=token_map1,
        use_template=use_template,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer1=teacher1_tokenizer,
        teacher_tokenizer2=teacher2_tokenizer,
    )



    train_dataset2 = PivotalTuningDatasetCapationPromptOnly(
        token_map=token_map2,
        use_template=use_template,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer1=teacher1_tokenizer,
        teacher_tokenizer2=teacher2_tokenizer,
    )

    train_dataset1.blur_amount = 200
    train_dataset2.blur_amount = 200

    train_dataloader1 = text2img_dataloader_noise(
            train_dataset=train_dataset1,
            train_batch_size=train_batch_size,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer1=teacher1_tokenizer,
            teacher_tokenizer2=teacher2_tokenizer,
            cached_latents=cached_latents,
        )

    train_dataloader2 = text2img_dataloader_noise(
            train_dataset=train_dataset2,
            train_batch_size=train_batch_size,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer1=teacher1_tokenizer,
            teacher_tokenizer2=teacher2_tokenizer,
            cached_latents=cached_latents,
        )


    index_no_updates1 = torch.arange(len(student_tokenizer)) != -1
    index_no_updates2 = torch.arange(len(student_tokenizer)) != -1

    for tok_id in placeholder_token_ids1:
        index_no_updates1[tok_id] = False

    for tok_id in placeholder_token_ids2:
        index_no_updates2[tok_id] = False

    student_unet.requires_grad_(False)
    student_vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        student_text_encoder.text_model.encoder.parameters(),
        student_text_encoder.text_model.final_layer_norm.parameters(),
        student_text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    if cached_latents:
        student_vae = None


    if not use_extended_lora:
        unet_lora_params, _ = inject_trainable_lora(
            student_unet,
            r=lora_rank,
            target_replace_module=lora_unet_target_modules,
            dropout_p=lora_dropout_p,
            scale=lora_scale,
        )
    else:
        print("PTI : USING EXTENDED UNET!!!")
        lora_unet_target_modules = (
            lora_unet_target_modules | UNET_EXTENDED_TARGET_REPLACE
        )
        print("PTI : Will replace modules: ", lora_unet_target_modules)

        unet_lora_params, _ = inject_trainable_lora_extended(
            student_unet, r=lora_rank, target_replace_module=lora_unet_target_modules
        )
    print(f"PTI : has {len(unet_lora_params)} lora")

    print("PTI : Before training:")
    inspect_lora(student_unet)

    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
    ]

    student_text_encoder.requires_grad_(False)

    if continue_inversion:
        params_to_optimize += [
            {
                "params": student_text_encoder.get_input_embeddings().parameters(),
                "lr": continue_inversion_lr
                if continue_inversion_lr is not None
                else ti_lr,
            }
        ]
        student_text_encoder.requires_grad_(True)
        params_to_freeze = itertools.chain(
            student_text_encoder.text_model.encoder.parameters(),
            student_text_encoder.text_model.final_layer_norm.parameters(),
            student_text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        for param in params_to_freeze:
            param.requires_grad = False
    else:
        student_text_encoder.requires_grad_(False)
    if train_text_encoder:
        student_text_encoder_lora_params, _ = inject_trainable_lora(
            student_text_encoder,
            target_replace_module=lora_clip_target_modules,
            r=lora_rank,
        )
        params_to_optimize += [
            {
                "params": itertools.chain(*student_text_encoder_lora_params),
                "lr": text_encoder_lr,
            }
        ]
        inspect_lora(student_text_encoder)

    print(params_to_optimize)

    lora_optimizers = optim.AdamW(params_to_optimize, weight_decay=weight_decay_lora)

    student_unet.train()
    if train_text_encoder:
        student_text_encoder.train()

    train_dataset1.blur_amount = 70
    train_dataset2.blur_amount = 70

    lr_scheduler_lora = get_scheduler(
        lr_scheduler_lora,
        optimizer=lora_optimizers,
        num_warmup_steps=lr_warmup_steps_lora,
        num_training_steps=max_train_steps_tuning,
    )

    placeholder_token_ids = placeholder_token_ids1 + placeholder_token_ids2
    placeholder_tokens = placeholder_tokens1 + placeholder_tokens2

    perform_tuning(
        teacher1_unet=teacher1_unet,
        teacher2_unet=teacher2_unet,
        teacher1_text_encoder=teacher1_text_encoder,
        teacher2_text_encoder=teacher2_text_encoder,
        student_unet=student_unet,
        vae=student_vae,
        student_text_encoder=student_text_encoder,
        dataloader1=train_dataloader1,
        dataloader2=train_dataloader2,
        num_steps=max_train_steps_tuning,
        cached_latents=cached_latents,
        scheduler=noise_scheduler,
        optimizer=lora_optimizers,
        save_steps=save_steps,
        placeholder_tokens=placeholder_tokens,
        placeholder_token_ids=placeholder_token_ids,
        save_path=output_dir,
        lr_scheduler_lora=lr_scheduler_lora,
        lora_unet_target_modules=lora_unet_target_modules,
        lora_clip_target_modules=lora_clip_target_modules,
        mask_temperature=mask_temperature,
        tokenizer=student_tokenizer,
        out_name=out_name,

        log_wandb=log_wandb,
        wandb_log_prompt_cnt=wandb_log_prompt_cnt,
        class_token=class_token,
        train_inpainting=train_inpainting,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training script for LoRA model')


    parser.add_argument('--lora_path1', type=str, required=True, help='Path to first LoRA model')
    parser.add_argument('--lora_path2', type=str, required=True, help='Path to second LoRA model')
    parser.add_argument('--instance_data_dir', type=str, required=False, default="", help='Directory containing instance data')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for saving models')


    parser.add_argument('--train_text_encoder', action='store_true', default=True, help='Whether to train text encoder')
    parser.add_argument('--pretrained_vae_name_or_path', type=str, default=None, help='Path to pretrained VAE')
    parser.add_argument('--revision', type=str, default=None, help='Revision of pretrained model')
    parser.add_argument('--perform_inversion', action='store_true', default=True, help='Perform inversion')
    parser.add_argument('--use_template', type=str, choices=[None, 'object', 'style'], default=None, help='Template to use')
    parser.add_argument('--train_inpainting', action='store_true', default=False, help='Train for inpainting')
    parser.add_argument('--placeholder_tokens', type=str, default='', help='Placeholder tokens')
    parser.add_argument('--placeholder_tokens1', type=str, default='', help='First set of placeholder tokens')
    parser.add_argument('--placeholder_tokens2', type=str, default='', help='Second set of placeholder tokens')
    parser.add_argument('--placeholder_token_at_data', type=str, default=None, help='Placeholder token at data')
    parser.add_argument('--initializer_tokens', type=str, default=None, help='Initializer tokens')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution for training')
    parser.add_argument('--color_jitter', action='store_true', default=True, help='Apply color jitter')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--sample_batch_size', type=int, default=1, help='Sampling batch size')
    parser.add_argument('--max_train_steps_tuning', type=int, default=1000, help='Maximum number of tuning steps')
    parser.add_argument('--max_train_steps_ti', type=int, default=1000, help='Maximum number of text inversion steps')
    parser.add_argument('--save_steps', type=int, default=100, help='Steps between saving checkpoints')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False, help='Enable gradient checkpointing')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank')

    parser.add_argument('--lora_unet_target_modules', type=str, default={"CrossAttention","Attention","GEGLU"},
                       help='Target modules for UNet LoRA (comma-separated)')
    parser.add_argument('--lora_clip_target_modules', type=str, default={"CLIPSdpaAttention"},
                       help='Target modules for CLIP LoRA (comma-separated)')
    parser.add_argument('--lora_dropout_p', type=float, default=0.0, help='LoRA dropout probability')
    parser.add_argument('--lora_scale', type=float, default=1.0, help='LoRA scale factor')
    parser.add_argument('--use_extended_lora', action='store_true', default=False, help='Use extended LoRA')
    parser.add_argument('--clip_ti_decay', action='store_true', default=True, help='Enable CLIP TI decay')
    parser.add_argument('--learning_rate_unet', type=float, default=1e-4, help='Learning rate for UNet')
    parser.add_argument('--learning_rate_text', type=float, default=1e-5, help='Learning rate for text encoder')
    parser.add_argument('--learning_rate_ti', type=float, default=5e-4, help='Learning rate for text inversion')
    parser.add_argument('--continue_inversion', action='store_true', default=True, help='Continue inversion')
    parser.add_argument('--continue_inversion_lr', type=float, default=None, help='Learning rate for continued inversion')
    parser.add_argument('--use_face_segmentation_condition', action='store_true', default=False, help='Use face segmentation condition')
    parser.add_argument('--cached_latents', action='store_true', default=True, help='Use cached latents')
    parser.add_argument('--use_mask_captioned_data', action='store_true', default=False, help='Use mask captioned data')
    parser.add_argument('--mask_temperature', type=float, default=1.0, help='Mask temperature')
    parser.add_argument('--scale_lr', action='store_true', default=False, help='Scale learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='linear', help='Learning rate scheduler')
    parser.add_argument('--lr_warmup_steps', type=int, default=0, help='Learning rate warmup steps')
    parser.add_argument('--lr_scheduler_lora', type=str, default='linear', help='LoRA learning rate scheduler')
    parser.add_argument('--lr_warmup_steps_lora', type=int, default=0, help='LoRA learning rate warmup steps')
    parser.add_argument('--weight_decay_ti', type=float, default=0.00, help='Weight decay for text inversion')
    parser.add_argument('--weight_decay_lora', type=float, default=0.001, help='Weight decay for LoRA')
    parser.add_argument('--use_8bit_adam', action='store_true', default=False, help='Use 8-bit Adam optimizer')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--log_wandb', action='store_true', default=False, help='Log to Weights & Biases')
    parser.add_argument('--wandb_log_prompt_cnt', type=int, default=10, help='Number of prompts to log to W&B')
    parser.add_argument('--wandb_project_name', type=str, default='new_pti_project', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='new_pti_entity', help='W&B entity')
    parser.add_argument('--proxy_token', type=str, default='person', help='Proxy token')
    parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true', default=False,
                       help='Enable xformers memory efficient attention')
    parser.add_argument('--out_name', type=str, default='final_lora', help='Name for the output model')

    args = parser.parse_args()


    args_dict = vars(args)

    train(**args_dict)