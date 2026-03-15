import torch
from tqdm.auto import tqdm
import numpy as np
from PIL import Image

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

def init_from_pointe(input_path, use_image=False):
    """
    Initialize a point cloud from text or an image
    
    Args:
        input_path (str): text prompt or path to an image file
        use_image (bool): whether to use an image instead of text as input
    
    Returns:
        xyz (numpy.ndarray): point cloud coordinates
        rgb (numpy.ndarray): point cloud colors
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_image:
        # Image-to-point-cloud configuration
        print('Creating image-to-point-cloud model...')
        base_name = 'base300M'  # use the image-based model
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
        
        print('Creating upsampler model...')
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
        
        print('Downloading image-to-point-cloud model checkpoint...')
        base_model.load_state_dict(load_checkpoint(base_name, device))
        print('Downloading upsampler model checkpoint...')
        upsampler_model.load_state_dict(load_checkpoint('upsample', device))
        
        # Configure sampler for image conditioning
        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 3.0],
        )

        img = Image.open('/root/lora_train_imgs/lora_red_light_diode_front/6.jpg')
        
        # Generate samples from the model
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
            samples = x
            
    else:
        # Original text-to-point-cloud configuration
        print('Creating text-to-point-cloud base model...')
        base_name = 'base40M-textvec'
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
        
        print('Creating upsampler model...')
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
        
        print('Downloading text-to-point-cloud model checkpoint...')
        base_model.load_state_dict(load_checkpoint(base_name, device))
        print('Downloading upsampler model checkpoint...')
        upsampler_model.load_state_dict(load_checkpoint('upsample', device))
        
        # Configure sampler for text conditioning
        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=('texts', ''),  # use texts as conditioning
        )
        
        # Generate samples from the model
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[input_path]))):
            samples = x
    
    # Convert output to point cloud
    pc = sampler.output_to_point_clouds(samples)[0]
    xyz = pc.coords
    rgb = np.zeros_like(xyz)
    rgb[:,0], rgb[:,1], rgb[:,2] = pc.channels['R'], pc.channels['G'], pc.channels['B']
    
    return xyz, rgb