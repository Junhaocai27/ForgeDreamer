import random
import imageio
import os
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GenerateCamParams, GuidanceParams
import math
from torchvision.utils import save_image
import torchvision.transforms as T

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def adjust_text_embeddings(embeddings, azimuth, guidance_opt):
    text_z_list = []
    weights_list = []
    K = 0

    if guidance_opt.fixed_view:
        text_z_, weights_ = get_pos_neg_text_embeddings_fixed(embeddings, azimuth, guidance_opt)
    else:
        text_z_, weights_ = get_pos_neg_text_embeddings_rand(embeddings, azimuth, guidance_opt)
    
    K = max(K, weights_.shape[0])
    text_z_list.append(text_z_)
    weights_list.append(weights_)

    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0)

    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0)
    return text_embeddings, weights

def get_pos_neg_text_embeddings_rand(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']

        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']

        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def get_pos_neg_text_embeddings_fixed(embeddings, azimuth_val, opt):
    device = embeddings['front'].device

    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
            
        r = torch.tensor(float(r), device=device, dtype=torch.float16)
        start_z = embeddings['front']
        end_z = embeddings['side']

        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        
        r_val = r.item()
        if r_val > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r_val * opt.front_decay_factor) * opt.negative_w
        if r_val < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r_val) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w], device=device)
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        
        r = torch.tensor(float(r), device=device, dtype=torch.float16)
        start_z = embeddings['side']
        end_z = embeddings['back']
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        
        r_val = r.item()
        front_neg_w = opt.negative_w 
        if r_val > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r_val * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w], device=device)
    
    return text_z, weights.to(text_z.device)

def prepare_embeddings(guidance_opt, guidance):
    embeddings = {}
    embeddings['default'] = guidance.get_text_embeds([guidance_opt.text])
    embeddings['uncond'] = guidance.get_text_embeds([guidance_opt.negative])

    for d in ['front', 'side', 'back']:
        embeddings[d] = guidance.get_text_embeds([f"{guidance_opt.text}, {d} view"])
    embeddings['inverse_text'] = guidance.get_text_embeds(guidance_opt.inverse_text)
    return embeddings

def prepare_embeddings_with_view_triggers(guidance_opt, guidance):
    """
    Automatically identifies <> triggers and generates view-specific text embeddings.
    """
    base_text = guidance_opt.text
    trigger_pattern = r'<([^>]+)>'
    base_triggers = re.findall(trigger_pattern, base_text)
    
    if not base_triggers:
        print("Warning: No trigger word wrapped in <> found in the text, using the original text.")
        base_trigger = None
    else:
        base_trigger = base_triggers[0]
        print(f"✅ Base trigger detected: <{base_trigger}>")
    
    if base_trigger:
        front_text = base_text.replace(f"<{base_trigger}>", f"<{base_trigger}_front>")
        up_text = base_text.replace(f"<{base_trigger}>", f"<{base_trigger}_up>")
    else:
        front_text = base_text + ", front view"
        up_text = base_text + ", up view"
    
    print("\n==== View-specific Text Prompts ====")
    print(f"Original text: {base_text}")
    print(f"Front view text: {front_text}")  
    print(f"Up view text: {up_text}")
    print("============================\n")
    
    embeddings = {}
    
    embeddings_front = {}
    embeddings_front['default'] = guidance.get_text_embeds([front_text])
    embeddings_front['uncond'] = guidance.get_text_embeds([guidance_opt.negative])
    
    for d in ['front', 'side', 'back']:
        view_text = f"{front_text}, {d} view"
        embeddings_front[d] = guidance.get_text_embeds([view_text])
    
    if isinstance(guidance_opt.inverse_text, list):
        inverse_texts_front = []
        for inv_text in guidance_opt.inverse_text:
            if base_trigger:
                inv_front = inv_text.replace(f"<{base_trigger}>", f"<{base_trigger}_front>")
            else:
                inv_front = inv_text + ", front view"
            inverse_texts_front.append(inv_front)
        embeddings_front['inverse_text'] = guidance.get_text_embeds(inverse_texts_front)
    else:
        if base_trigger:
            inv_front = guidance_opt.inverse_text.replace(f"<{base_trigger}>", f"<{base_trigger}_front>")
        else:
            inv_front = guidance_opt.inverse_text + ", front view"
        embeddings_front['inverse_text'] = guidance.get_text_embeds([inv_front])
    
    embeddings_up = {}
    embeddings_up['default'] = guidance.get_text_embeds([up_text])
    embeddings_up['uncond'] = guidance.get_text_embeds([guidance_opt.negative])
    
    for d in ['front', 'side', 'back']:
        view_text = f"{up_text}, {d} view"
        embeddings_up[d] = guidance.get_text_embeds([view_text])
    
    if isinstance(guidance_opt.inverse_text, list):
        inverse_texts_up = []
        for inv_text in guidance_opt.inverse_text:
            if base_trigger:
                inv_up = inv_text.replace(f"<{base_trigger}>", f"<{base_trigger}_up>")
            else:
                inv_up = inv_text + ", up view"
            inverse_texts_up.append(inv_up)
        embeddings_up['inverse_text'] = guidance.get_text_embeds(inverse_texts_up)
    else:
        if base_trigger:
            inv_up = guidance_opt.inverse_text.replace(f"<{base_trigger}>", f"<{base_trigger}_up>")
        else:
            inv_up = guidance_opt.inverse_text + ", up view"
        embeddings_up['inverse_text'] = guidance.get_text_embeds([inv_up])
    
    view_embeddings = {
        'front': embeddings_front,
        'up': embeddings_up
    }
    
    return view_embeddings

def select_embeddings_by_camera(view_embeddings, viewpoint_cam):
    """
    Selects appropriate embeddings based on the camera angle.
    """
    polar = getattr(viewpoint_cam, 'delta_polar', 0)
    
    if polar != 0:
        current_embeddings = view_embeddings['up']
        view_type = "up"
    else:
        current_embeddings = view_embeddings['front'] 
        view_type = "front"
    
    return current_embeddings, view_type

def guidance_setup_enhanced(guidance_opt):
    """
    Enhanced guidance setup with support for view-specific triggers.
    """
    if guidance_opt.guidance=="SD":
        from guidance.sd_utils_copy_original3 import StableDiffusion
        guidance = StableDiffusion(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O, 
                                   guidance_opt.t_range, guidance_opt.max_t_range, 
                                   num_train_timesteps=guidance_opt.num_train_timesteps, 
                                   ddim_inv=guidance_opt.ddim_inv,
                                   textual_inversion_path = guidance_opt.textual_inversion_path,
                                   LoRA_path = guidance_opt.LoRA_path,
                                   guidance_opt=guidance_opt)
    else:
        raise ValueError(f'{guidance_opt.guidance} not supported.')
    
    if guidance is not None:
        for p in guidance.parameters():
            p.requires_grad = False
    
    view_embeddings = prepare_embeddings_with_view_triggers(guidance_opt, guidance)
    
    return guidance, view_embeddings

def guidance_setup(guidance_opt):
    if guidance_opt.guidance=="SD":
        from guidance.sd_utils_copy_original3 import StableDiffusion
        guidance = StableDiffusion(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O, 
                                   guidance_opt.t_range, guidance_opt.max_t_range, 
                                   num_train_timesteps=guidance_opt.num_train_timesteps, 
                                   ddim_inv=guidance_opt.ddim_inv,
                                   textual_inversion_path = guidance_opt.textual_inversion_path,
                                   LoRA_path = guidance_opt.LoRA_path,
                                   guidance_opt=guidance_opt)
    else:
        raise ValueError(f'{guidance_opt.guidance} not supported.')
    if guidance is not None:
        for p in guidance.parameters():
            p.requires_grad = False
    embeddings = prepare_embeddings(guidance_opt, guidance)
    return guidance, embeddings


def training(dataset, opt, pipe, gcams, guidance_opt, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_video):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gcams, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    save_folder = os.path.join(dataset._model_path,"train_process/")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('train_process is in :', save_folder)
    
    use_control_net = False
    
    guidance, view_embeddings = guidance_setup_enhanced(guidance_opt)   
    
    viewpoint_stack = None
    viewpoint_stack_around = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if opt.save_process:
        save_folder_proc = os.path.join(scene.args._model_path,"process_videos/")
        if not os.path.exists(save_folder_proc):
            os.makedirs(save_folder_proc)
        process_view_points = scene.getCircleVideoCameras(batch_size=opt.pro_frames_num,render45=opt.pro_render_45).copy()    
        save_process_iter = opt.iterations // len(process_view_points)
        pro_img_frames = []
    
    text_z_inverse_front = torch.cat([view_embeddings['front']['uncond'], view_embeddings['front']['inverse_text']], dim=0)
    text_z_inverse_up = torch.cat([view_embeddings['up']['uncond'], view_embeddings['up']['inverse_text']], dim=0)

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, guidance_opt.text)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.update_feature_learning_rate(iteration)
        gaussians.update_rotation_learning_rate(iteration)
        gaussians.update_scaling_learning_rate(iteration)
        
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        if not opt.use_progressive:                
            if iteration >= opt.progressive_view_iter and iteration % opt.scale_up_cameras_iter == 0:
                scene.pose_args.fovy_range[0] = max(scene.pose_args.max_fovy_range[0], scene.pose_args.fovy_range[0] * opt.fovy_scale_up_factor[0])
                scene.pose_args.fovy_range[1] = min(scene.pose_args.max_fovy_range[1], scene.pose_args.fovy_range[1] * opt.fovy_scale_up_factor[1])

                scene.pose_args.radius_range[1] = max(scene.pose_args.max_radius_range[1], scene.pose_args.radius_range[1] * opt.scale_up_factor)
                scene.pose_args.radius_range[0] = max(scene.pose_args.max_radius_range[0], scene.pose_args.radius_range[0] * opt.scale_up_factor)

                scene.pose_args.theta_range[1] = min(scene.pose_args.max_theta_range[1], scene.pose_args.theta_range[1] * opt.phi_scale_up_factor)
                scene.pose_args.theta_range[0] = max(scene.pose_args.max_theta_range[0], scene.pose_args.theta_range[0] * 1/opt.phi_scale_up_factor)

                scene.pose_args.phi_range[0] = max(scene.pose_args.max_phi_range[0] , scene.pose_args.phi_range[0] * opt.phi_scale_up_factor)
                scene.pose_args.phi_range[1] = min(scene.pose_args.max_phi_range[1], scene.pose_args.phi_range[1] * opt.phi_scale_up_factor)
                
                print('scale up theta_range to:', scene.pose_args.theta_range)
                print('scale up radius_range to:', scene.pose_args.radius_range)
                print('scale up phi_range to:', scene.pose_args.phi_range)
                print('scale up fovy_range to:', scene.pose_args.fovy_range)

        if not viewpoint_stack and guidance_opt.fixed_view:
            if iteration < opt.warmup_iter:
                viewpoint_stack = scene.getCircleVideoCameras(render45=False).copy()
                render45_mode = False
            else:
                viewpoint_stack = scene.getCircleVideoCameras(render45=True).copy()
                render45_mode = True
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getRandTrainCameras().copy()

        C_batch_size = guidance_opt.C_batch_size
        viewpoint_cams = []
        images = []
        text_z_ = []
        weights_ = []
        depths = []
        alphas = []
        scales = []

        for i in range(C_batch_size):
            try:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))            
            except:
                if guidance_opt.fixed_view:
                    if iteration < opt.warmup_iter:
                        viewpoint_stack = scene.getCircleVideoCameras(render45=False).copy()
                        render45_mode = False
                    else:
                        viewpoint_stack = scene.getCircleVideoCameras(render45=True).copy()
                        render45_mode = True
                else:
                    viewpoint_stack = scene.getRandTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            current_embeddings, view_type = select_embeddings_by_camera(view_embeddings, viewpoint_cam)
            
            azimuth = viewpoint_cam.delta_azimuth
            text_z = [current_embeddings['uncond']]

            if guidance_opt.perpneg:
                text_z_comp, weights = adjust_text_embeddings(current_embeddings, azimuth, guidance_opt)
                text_z.append(text_z_comp)
                weights_.append(weights)
            else:                
                if azimuth >= -90 and azimuth < 90:
                    if azimuth >= 0:
                        r = 1 - azimuth / 90
                    else:
                        r = 1 + azimuth / 90
                    start_z = current_embeddings['front']
                    end_z = current_embeddings['side']
                else:
                    if azimuth >= 0:
                        r = 1 - (azimuth - 90) / 90
                    else:
                        r = 1 + (azimuth + 90) / 90
                    start_z = current_embeddings['side']
                    end_z = current_embeddings['back']
                text_z.append(r * start_z + (1 - r) * end_z)

            text_z = torch.cat(text_z, dim=0)
            text_z_.append(text_z)

            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, 
                                sh_deg_aug_ratio = dataset.sh_deg_aug_ratio, 
                                bg_aug_ratio = 0.0, 
                                shs_aug_ratio = dataset.shs_aug_ratio, 
                                scale_aug_ratio = dataset.scale_aug_ratio)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]

            scales.append(render_pkg["scales"])
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)
            viewpoint_cams.append(viewpoint_cam)

        images = torch.stack(images, dim=0)
        depths = torch.stack(depths, dim=0)
        alphas = torch.stack(alphas, dim=0)

        warm_up_rate = 1. - min(iteration/opt.warmup_iter,1.)
        guidance_scale = guidance_opt.guidance_scale
        _aslatent = False
        if iteration < opt.geo_iter or random.random()< opt.as_latent_ratio:
            _aslatent=True
        if iteration > opt.use_control_net_iter and (random.random() < guidance_opt.controlnet_ratio):
            use_control_net = True
            
        first_cam_embeddings, first_view_type = select_embeddings_by_camera(view_embeddings, viewpoint_cams[0])
        text_z_inverse = text_z_inverse_front if first_view_type == "front" else text_z_inverse_up
            
        if guidance_opt.perpneg:
            loss = guidance.train_step_perpneg(torch.stack(text_z_, dim=1), images, 
                                                pred_depth=depths, pred_alpha=alphas,
                                                grad_scale=guidance_opt.lambda_guidance,
                                                use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                                weights = torch.stack(weights_, dim=1), resolution=(gcams.image_h, gcams.image_w),
                                                guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse, opt=opt)
            
        else:
            loss = guidance.train_step(torch.stack(text_z_, dim=1), images, 
                                    pred_depth=depths, pred_alpha=alphas,
                                    grad_scale=guidance_opt.lambda_guidance,
                                    use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                    resolution=(gcams.image_h, gcams.image_w),
                                    guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse)

        scales = torch.stack(scales, dim=0)

        loss_scale = torch.mean(scales,dim=-1).mean()
        loss_tv = tv_loss(images) + tv_loss(depths) 

        loss = loss + opt.lambda_tv * loss_tv + opt.lambda_scale * loss_scale
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if opt.save_process:
                if iteration % save_process_iter == 0 and len(process_view_points) > 0:
                    viewpoint_cam_p = process_view_points.pop(0)
                    render_p = render(viewpoint_cam_p, gaussians, pipe, background, test=True)
                    img_p = torch.clamp(render_p["render"], 0.0, 1.0) 
                    img_p = img_p.detach().cpu().permute(1,2,0).numpy()
                    img_p = (img_p * 255).round().astype('uint8')
                    pro_img_frames.append(img_p)  

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in testing_iterations):
                if save_video:
                    video_inference(iteration, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene._model_path + "/chkpnt" + str(iteration) + ".pth")

    if opt.save_process:
        imageio.mimwrite(os.path.join(save_folder_proc, "video_rgb.mp4"), pro_img_frames, fps=30, quality=8)

def prepare_output_and_logger(args):    
    if not args._model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args._model_path = os.path.join("./output/", args.workspace)
        
    print("Output folder: {}".format(args._model_path))
    os.makedirs(args._model_path, exist_ok = True)

    if args.opt_path is not None:
        os.system(' '.join(['cp', args.opt_path, os.path.join(args._model_path, 'config.yaml')]))

    with open(os.path.join(args._model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args._model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        save_folder = os.path.join(scene.args._model_path,"test_six_views/{}_iteration".format(iteration))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print('Test views are in:', save_folder)
        torch.cuda.empty_cache()
        config = ({'name': 'test', 'cameras' : scene.getTestCameras()})
        if config['cameras'] and len(config['cameras']) > 0:
            for idx, viewpoint in enumerate(config['cameras']):
                render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
                rgb, depth = render_out["render"],render_out["depth"]
                if depth is not None:
                    depth_norm = depth/depth.max()
                    save_image(depth_norm,os.path.join(save_folder,"render_depth_{}.png".format(viewpoint.uid)))

                image = torch.clamp(rgb, 0.0, 1.0)
                save_image(image,os.path.join(save_folder,"render_view_{}.png".format(viewpoint.uid)))
                if tb_writer:
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.uid), image[None], global_step=iteration)     
            print("\n[ITER {}] Eval Done!".format(iteration))
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def video_inference(iteration, scene : Scene, renderFunc, renderArgs):
    sharp = T.RandomAdjustSharpness(3, p=1.0)

    save_folder = os.path.join(scene.args._model_path,"videos/{}_iteration".format(iteration))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('Videos are in:', save_folder)
    torch.cuda.empty_cache()
    config = ({'name': 'test', 'cameras' : scene.getCircleVideoCameras()})
    if config['cameras'] and len(config['cameras']) > 0:
        img_frames = []
        depth_frames = []
        print("Generating Video using", len(config['cameras']), "different view points")
        for idx, viewpoint in enumerate(config['cameras']):
            render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
            rgb,depth = render_out["render"],render_out["depth"]
            if depth is not None:
                depth_norm = depth/depth.max()
                depths = torch.clamp(depth_norm, 0.0, 1.0) 
                depths = depths.detach().cpu().permute(1,2,0).numpy()
                depths = (depths * 255).round().astype('uint8')          
                depth_frames.append(depths)    
  
            image = torch.clamp(rgb, 0.0, 1.0) 
            image = image.detach().cpu().permute(1,2,0).numpy()
            image = (image * 255).round().astype('uint8')
            img_frames.append(image)    
        imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), img_frames, fps=30, quality=8)
        if len(depth_frames) > 0:
            imageio.mimwrite(os.path.join(save_folder, "video_depth_{}.mp4".format(iteration)), depth_frames, fps=30, quality=8)
        print("\n[ITER {}] Video Save Done!".format(iteration))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import yaml

    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--opt', type=str, default='/home/s414e2/CJH/Text-to-3D/LucidDreamer/configs/screw_lora.yaml')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_ratio", type=int, default=5)
    parser.add_argument("--save_ratio", type=int, default=2)
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gcp = GenerateCamParams(parser)
    gp = GuidanceParams(parser)

    args = parser.parse_args(sys.argv[1:])

    if args.opt is not None:
        with open(args.opt) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
        lp.load_yaml(opts.get('ModelParams', None))
        op.load_yaml(opts.get('OptimizationParams', None))
        pp.load_yaml(opts.get('PipelineParams', None))
        gcp.load_yaml(opts.get('GenerateCamParams', None))
        gp.load_yaml(opts.get('GuidanceParams', None))
        
        lp.opt_path = args.opt
        args.port = opts['port']
        args.save_video = opts.get('save_video', True)
        args.seed = opts.get('seed', 0)
        args.device = opts.get('device', 'cuda')

        gp.g_device = args.device
        lp.data_device = args.device
        gcp.device = args.device

    test_iter = [1] + [k * op.iterations // args.test_ratio for k in range(1, args.test_ratio)] + [op.iterations]
    args.test_iterations = test_iter

    save_iter = [k * op.iterations // args.save_ratio for k in range(1, args.save_ratio)] + [op.iterations]
    args.save_iterations = save_iter

    print('Test iter:', args.test_iterations)
    print('Save iter:', args.save_iterations)

    print("Optimizing " + lp._model_path)

    safe_state(args.quiet, seed=args.seed)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp, op, pp, gcp, gp, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.save_video)

    print("\nTraining complete.")
