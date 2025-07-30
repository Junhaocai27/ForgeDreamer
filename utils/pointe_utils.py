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
    从文本或图像初始化点云
    
    参数:
        input_path (str): 文本提示或图像文件的路径
        use_image (bool): 是否使用图像而不是文本作为输入
    
    返回:
        xyz (numpy.ndarray): 点云坐标
        rgb (numpy.ndarray): 点云颜色
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_image:
        # 图像到点云的配置
        print('创建图像到点云模型...')
        base_name = 'base300M' # 使用基于图像的模型
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
        
        print('创建上采样模型...')
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
        
        print('下载图像到点云模型检查点...')
        base_model.load_state_dict(load_checkpoint(base_name, device))
        print('下载上采样模型检查点...')
        upsampler_model.load_state_dict(load_checkpoint('upsample', device))
        
        # 配置采样器用于图像条件
        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 3.0],
        )

        img = Image.open('/root/lora_train_imgs/lora_red_light_diode_front/6.jpg')
        
        # 从模型生成样本
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
            samples = x
            
    else:
        # 原始的文本到点云配置
        print('创建文本到点云基础模型...')
        base_name = 'base40M-textvec'
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
        
        print('创建上采样模型...')
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
        
        print('下载文本到点云模型检查点...')
        base_model.load_state_dict(load_checkpoint(base_name, device))
        print('下载上采样模型检查点...')
        upsampler_model.load_state_dict(load_checkpoint('upsample', device))
        
        # 配置采样器用于文本条件
        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=('texts', ''),  # 使用texts作为条件
        )
        
        # 从模型生成样本
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[input_path]))):
            samples = x
    
    # 转换输出为点云
    pc = sampler.output_to_point_clouds(samples)[0]
    xyz = pc.coords
    rgb = np.zeros_like(xyz)
    rgb[:,0], rgb[:,1], rgb[:,2] = pc.channels['R'], pc.channels['G'], pc.channels['B']
    
    return xyz, rgb