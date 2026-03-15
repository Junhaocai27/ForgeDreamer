#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
import random
import torch.nn.functional as F
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from utils.pointe_utils import init_from_pointe
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.general_utils import inverse_sigmoid_np
from scene.gaussian_model import BasicPointCloud


class RandCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int 
    delta_polar : np.array
    delta_azimuth : np.array
    delta_radius : np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


class RSceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    test_cameras: list
    ply_path: str

# def getNerfppNorm(cam_info):
#     def get_center_and_diag(cam_centers):
#         cam_centers = np.hstack(cam_centers)
#         avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
#         center = avg_cam_center
#         dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
#         diagonal = np.max(dist)
#         return center.flatten(), diagonal

#     cam_centers = []

#     for cam in cam_info:
#         W2C = getWorld2View2(cam.R, cam.T)
#         C2W = np.linalg.inv(W2C)
#         cam_centers.append(C2W[:3, 3:4])

#     center, diagonal = get_center_and_diag(cam_centers)
#     radius = diagonal * 1.1

#     translate = -center

#     return {"translate": translate, "radius": radius}



def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

#only test_camera
def readCircleCamInfo(path,opt):
    print("Reading Test Transforms")
    test_cam_infos = GenerateCircleCameras(opt,render45 = opt.render_45)
    ply_path = os.path.join(path, "init_points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = opt.init_num_pts       
        if opt.init_shape == 'sphere':
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi        
            radius = np.random.rand(num_pts)*0.5
            # We create random points inside the bounds of sphere
            xyz = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1) # [B, 3]
        elif opt.init_shape == 'box':
            xyz = np.random.random((num_pts, 3)) * 1.0 - 0.5
        elif opt.init_shape == 'rectangle_x':
            xyz = np.random.random((num_pts, 3))
            xyz[:, 0] = xyz[:, 0] * 0.6 - 0.3
            xyz[:, 1] = xyz[:, 1] * 1.2 - 0.6
            xyz[:, 2] = xyz[:, 2] * 0.5 - 0.25
        elif opt.init_shape == 'rectangle_z':
            xyz = np.random.random((num_pts, 3))
            xyz[:, 0] = xyz[:, 0] * 0.8 - 0.4
            xyz[:, 1] = xyz[:, 1] * 0.6 - 0.3
            xyz[:, 2] = xyz[:, 2] * 1.2 - 0.6
        elif opt.init_shape == 'pointe':
            num_pts = int(num_pts/5000)
            xyz,rgb = init_from_pointe(opt.init_prompt, opt.use_image)
            xyz[:,1] = - xyz[:,1]
            xyz[:,2] = xyz[:,2] + 0.15
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi        
            radius = np.random.rand(num_pts)*0.05
            # We create random points inside the bounds of sphere
            xyz_ball = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1) # [B, 3]expend_dims
            rgb_ball = np.random.random((4096, num_pts, 3))*0.0001
            rgb = (np.expand_dims(rgb,axis=1)+rgb_ball).reshape(-1,3)
            xyz = (np.expand_dims(xyz,axis=1)+np.expand_dims(xyz_ball,axis=0)).reshape(-1,3)
            xyz = xyz * 1.
            num_pts = xyz.shape[0]
        elif opt.init_shape == 'pointe_img':
            num_pts = int(num_pts/5000)
            xyz,rgb = init_from_pointe(opt.init_prompt, opt.use_image)
            xyz[:,1] = - xyz[:,1]
            xyz[:,2] = xyz[:,2] + 0.15
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi        
            radius = np.random.rand(num_pts)*0.05
            # We create random points inside the bounds of sphere
            xyz_ball = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1) # [B, 3]expend_dims
            rgb_ball = np.random.random((4096, num_pts, 3))*0.0001
            rgb = (np.expand_dims(rgb,axis=1)+rgb_ball).reshape(-1,3)
            xyz = (np.expand_dims(xyz,axis=1)+np.expand_dims(xyz_ball,axis=0)).reshape(-1,3)
            xyz = xyz * 1.
            num_pts = xyz.shape[0]
        elif opt.init_shape == 'scene':
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi        
            radius = np.random.rand(num_pts) + opt.radius_range[-1]*3
            # We create random points inside the bounds of sphere
            xyz = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1) # [B, 3]
        else:
            raise NotImplementedError()
        print(f"Generating random point cloud ({num_pts})...")

        shs = np.random.random((num_pts, 3)) / 255.0

        if opt.init_shape == 'pointe' and opt.use_pointe_rgb:
            pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((num_pts, 3)))
            storePly(ply_path, xyz, rgb * 255)
        else:
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = RSceneInfo(point_cloud=pcd,
                           test_cameras=test_cam_infos,
                           ply_path=ply_path)
    return scene_info
#borrow from https://github.com/ashawkey/stable-dreamfusion

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

# def circle_poses(radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), angle_overhead=30, angle_front=60):

#     theta = theta / 180 * np.pi
#     phi = phi / 180 * np.pi
#     angle_overhead = angle_overhead / 180 * np.pi
#     angle_front = angle_front / 180 * np.pi

#     centers = torch.stack([
#         radius * torch.sin(theta) * torch.sin(phi),
#         radius * torch.cos(theta),
#         radius * torch.sin(theta) * torch.cos(phi),
#     ], dim=-1) # [B, 3]

#     # lookat
#     forward_vector = safe_normalize(centers)
#     up_vector = torch.FloatTensor([0, 1, 0]).unsqueeze(0).repeat(len(centers), 1)
#     right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
#     up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

#     poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
#     poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
#     poses[:, :3, 3] = centers

#     return poses.numpy()

def circle_poses(radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.sin(theta) * torch.cos(phi),
        radius * torch.cos(theta),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses.numpy()

def gen_random_pos(size, param_range, gamma=1):
    lower, higher = param_range[0], param_range[1]
    
    mid = lower + (higher - lower) * 0.5
    radius = (higher - lower) * 0.5

    rand_ = torch.rand(size) # 0, 1
    sign = torch.where(torch.rand(size) > 0.5, torch.ones(size) * -1., torch.ones(size))
    rand_ = sign * (rand_ ** gamma)          

    return (rand_ * radius) + mid


def rand_poses(size, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], angle_overhead=30, angle_front=60, uniform_sphere_rate=0.5, rand_cam_gamma=1):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    # radius = torch.rand(size) * (radius_range[1] - radius_range[0]) + radius_range[0]
    radius = gen_random_pos(size, radius_range)

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                torch.randn(size),
                torch.abs(torch.randn(size)),
                torch.randn(size),
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        # thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
        # phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
        # phis[phis < 0] += 2 * np.pi

        # centers = torch.stack([
        #     radius * torch.sin(thetas) * torch.sin(phis),
        #     radius * torch.cos(thetas),
        #     radius * torch.sin(thetas) * torch.cos(phis),
        # ], dim=-1) # [B, 3]
        # thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
        # phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
        thetas = gen_random_pos(size, theta_range, rand_cam_gamma)
        phis = gen_random_pos(size, phi_range, rand_cam_gamma)
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.sin(thetas) * torch.cos(phis),
            radius * torch.cos(thetas),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if opt.jitter_pose:
        jit_center = opt.jitter_center # 0.015  # was 0.2
        jit_target = opt.jitter_target
        centers += torch.rand_like(centers) * jit_center - jit_center/2.0
        targets += torch.randn_like(centers) * jit_target

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    #up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise) #forward_vector

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1) #up_vector
    poses[:, :3, 3] = centers


    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses.numpy(), thetas.numpy(), phis.numpy(), radius.numpy()

def GenerateCircleCameras(opt, size=8, render45 = False):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    #generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar])
        phis = torch.FloatTensor([(idx / size) * 360])
        radius = torch.FloatTensor([opt.default_radius])
        # random pose on the fly
        poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                        height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))  
    if render45:
        # print(f"[DEBUG] Generating 45-degree view cameras")
        for idx in range(size):
            # thetas = torch.FloatTensor([opt.default_polar*2//3])
            thetas = torch.FloatTensor([55])
            phis = torch.FloatTensor([(idx / size) * 360])
            radius = torch.FloatTensor([opt.default_radius])
            # print(f"[DEBUG] 45-degree camera {idx}: theta={thetas.item()}, phi={phis.item()}, radius={radius.item()}")
            # random pose on the fly
            poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
            FovY = fovy
            FovX = fov

            # delta polar/azimuth/radius to default view
            delta_polar = thetas - opt.default_polar
            delta_azimuth = phis - opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
            delta_radius = radius - opt.default_radius
            cam_infos.append(RandCameraInfo(uid=idx+size, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                            height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))         
    # print(f"[DEBUG] Final camera count: {size}")
    return cam_infos

# def GenerateRandomFixedAngleCameras(opt, size=2000, SSAA=True, angle_ratio=0.5):
#     """
#     Generate random cameras with only front and 45-degree viewing angles
    
#     Args:
#         opt: configuration options
#         size: number of cameras to generate
#         SSAA: whether to enable super-sampling anti-aliasing
#         angle_ratio: ratio of front vs 45-degree views, default 0.5 means equal split
    
#     Returns:
#         cam_infos: list of camera info
#     """
    
#     # Compute number of front and 45-degree view cameras
#     front_view_count = int(size * angle_ratio)
#     angle45_view_count = size - front_view_count
    
#     cam_infos = []
    
#     # Set image dimensions
#     if SSAA:
#         ssaa = opt.SSAA
#     else:
#         ssaa = 1

#     image_h = opt.image_h * ssaa
#     image_w = opt.image_w * ssaa
    
#     # Generate front-view cameras
#     for idx in range(front_view_count):
#         # Fix polar angle to default front view
#         thetas = torch.FloatTensor([opt.default_polar])
        
#         # Random azimuth angle
#         phis = torch.FloatTensor([random.random() * 360])
        
#         # Random radius
#         radius = gen_random_pos(1, opt.radius_range, opt.rand_cam_gamma)
        
#         # Random focal length
#         fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
        
#         # Generate camera pose
#         poses = circle_poses(
#             radius=radius, 
#             theta=thetas, 
#             phi=phis, 
#             angle_overhead=opt.angle_overhead, 
#             angle_front=opt.angle_front
#         )
        
#         # Compute camera matrix
#         matrix = np.linalg.inv(poses[0])
#         R = -np.transpose(matrix[:3,:3])
#         R[:,0] = -R[:,0]
#         T = -matrix[:3, 3]
        
#         # Compute field of view
#         fovy = focal2fov(fov2focal(fov, image_h), image_w)
#         FovY = fovy
#         FovX = fov
        
#         # Compute delta relative to default view
#         delta_polar = thetas - opt.default_polar
#         delta_azimuth = phis - opt.default_azimuth
#         delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
#         delta_radius = radius - opt.default_radius
        
#         cam_infos.append(RandCameraInfo(
#             uid=idx, 
#             R=R, T=T, FovY=FovY, FovX=FovX,
#             width=image_w, height=image_h, 
#             delta_polar=delta_polar[0], 
#             delta_azimuth=delta_azimuth[0], 
#             delta_radius=delta_radius[0]
#         ))
    
#     # Generate 45-degree view cameras
#     for idx in range(angle45_view_count):
#         # Fix polar angle to 45-degree view
#         thetas = torch.FloatTensor([opt.default_polar * 2 // 3])
        
#         # Random azimuth angle
#         phis = torch.FloatTensor([random.random() * 360])
        
#         # Random radius
#         radius = gen_random_pos(1, opt.radius_range, opt.rand_cam_gamma)
        
#         # Random focal length
#         fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
        
#         # Generate camera pose
#         poses = circle_poses(
#             radius=radius, 
#             theta=thetas, 
#             phi=phis, 
#             angle_overhead=opt.angle_overhead, 
#             angle_front=opt.angle_front
#         )
        
#         # Compute camera matrix
#         matrix = np.linalg.inv(poses[0])
#         R = -np.transpose(matrix[:3,:3])
#         R[:,0] = -R[:,0]
#         T = -matrix[:3, 3]
        
#         # Compute field of view
#         fovy = focal2fov(fov2focal(fov, image_h), image_w)
#         FovY = fovy
#         FovX = fov
        
#         # Compute delta relative to default view
#         delta_polar = thetas - opt.default_polar
#         delta_azimuth = phis - opt.default_azimuth
#         delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
#         delta_radius = radius - opt.default_radius
        
#         cam_infos.append(RandCameraInfo(
#             uid=front_view_count + idx, 
#             R=R, T=T, FovY=FovY, FovX=FovX,
#             width=image_w, height=image_h, 
#             delta_polar=delta_polar[0], 
#             delta_azimuth=delta_azimuth[0], 
#             delta_radius=delta_radius[0]
#         ))
    
#     # Shuffle camera order to avoid all front-view cameras in first half
#     random.shuffle(cam_infos)
    
#     # Re-assign uid for continuity
#     for i, cam_info in enumerate(cam_infos):
#         cam_infos[i] = cam_info._replace(uid=i)
    
#     print(f"[INFO] Generated {front_view_count} front-view cameras and {angle45_view_count} 45-degree cameras, total {size}")
    
#     return cam_infos

def GenerateRandomFixedAngleCameras(opt, size=2000, SSAA=True, random_ratio=True, 
                                   min_front_ratio=0.3, max_front_ratio=0.7,
                                   angle_jitter=True, max_jitter_degrees=1.0):
    """
    Generate random cameras with only front and 45-degree viewing angles, optionally with angle jitter
    
    Args:
        opt: configuration options
        size: number of cameras to generate
        SSAA: whether to enable super-sampling anti-aliasing
        random_ratio: whether to use a random ratio, default True
        min_front_ratio: minimum fraction of front-view cameras, default 0.3
        max_front_ratio: maximum fraction of front-view cameras, default 0.7
        angle_jitter: whether to add angle jitter, default True
        max_jitter_degrees: maximum jitter in degrees, default 5.0
    
    Returns:
        cam_infos: list of camera info
    """
    
    # Use random ratio or fixed ratio
    if random_ratio:
        # Randomly determine front-view fraction
        front_ratio = random.uniform(min_front_ratio, max_front_ratio)
    else:
        # Use fixed ratio (backward compatible)
        front_ratio = 0.5
    
    # Compute number of front and 45-degree view cameras
    front_view_count = int(size * front_ratio)
    angle45_view_count = size - front_view_count
    
    jitter_info = f"(jitter±{max_jitter_degrees}°)" if angle_jitter else "(fixed angles)"
    print(f"[INFO] Random split: {front_view_count} front-view cameras ({front_ratio:.1%}) and {angle45_view_count} 45-degree cameras ({1-front_ratio:.1%}) {jitter_info}")
    
    cam_infos = []
    
    # Set image dimensions
    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa
    
    # Generate front-view cameras
    for idx in range(front_view_count):
        # Base polar angle is default front view
        base_theta = opt.default_polar
        
        # Add angle jitter
        if angle_jitter:
            # Add small random offset to polar angle
            theta_jitter = random.uniform(-max_jitter_degrees, max_jitter_degrees)
            actual_theta = base_theta + theta_jitter
            # Clamp to valid range
            actual_theta = max(10, min(170, actual_theta))  # limit to 10-170 degrees
        else:
            actual_theta = base_theta
            
        thetas = torch.FloatTensor([actual_theta])
        
        # Random azimuth
        base_phi = random.random() * 360
        if angle_jitter:
            # Also add small offset to azimuth (optional)
            phi_jitter = random.uniform(-max_jitter_degrees, max_jitter_degrees)
            actual_phi = (base_phi + phi_jitter) % 360
        else:
            actual_phi = base_phi
            
        phis = torch.FloatTensor([actual_phi])
        
        # Random radius
        radius = gen_random_pos(1, opt.radius_range, opt.rand_cam_gamma)
        
        # Random focal length
        fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
        
        # Generate camera pose
        poses = circle_poses(
            radius=radius, 
            theta=thetas, 
            phi=phis, 
            angle_overhead=opt.angle_overhead, 
            angle_front=opt.angle_front
        )
        
        # Compute camera matrix
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        
        # Compute field of view
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov
        
        # Compute delta relative to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        
        cam_infos.append(RandCameraInfo(
            uid=idx, 
            R=R, T=T, FovY=FovY, FovX=FovX,
            width=image_w, height=image_h, 
            delta_polar=delta_polar[0], 
            delta_azimuth=delta_azimuth[0], 
            delta_radius=delta_radius[0]
        ))
    
    # Generate 45-degree view cameras
    for idx in range(angle45_view_count):
        # Base 45-degree view
        base_theta = 55.0
        
        # Add angle jitter
        if angle_jitter:
            # Add small random offset to polar angle
            theta_jitter = random.uniform(-max_jitter_degrees, max_jitter_degrees)
            actual_theta = base_theta + theta_jitter
            # Clamp to valid range
            actual_theta = max(10, min(80, actual_theta))  # limit to 10-80 degrees, avoid getting too close to top view
        else:
            actual_theta = base_theta
            
        thetas = torch.FloatTensor([actual_theta])
        
        if angle_jitter:
            print(f"[DEBUG] 45-degree camera {idx}: theta={actual_theta:.1f}deg (base 55° + jitter {actual_theta-55:.1f}°)")
        else:
            print(f"[DEBUG] 45-degree camera {idx}: theta={actual_theta:.1f}deg")
        
        # Random azimuth
        base_phi = random.random() * 360
        if angle_jitter:
            # Also add small offset to azimuth
            phi_jitter = random.uniform(-max_jitter_degrees, max_jitter_degrees)
            actual_phi = (base_phi + phi_jitter) % 360
        else:
            actual_phi = base_phi
            
        phis = torch.FloatTensor([actual_phi])
        
        # Random radius
        radius = gen_random_pos(1, opt.radius_range, opt.rand_cam_gamma)
        
        # Random focal length
        fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
        
        # Generate camera pose
        poses = circle_poses(
            radius=radius, 
            theta=thetas, 
            phi=phis, 
            angle_overhead=opt.angle_overhead, 
            angle_front=opt.angle_front
        )
        
        # Compute camera matrix
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        
        # Compute field of view
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov
        
        # Compute delta relative to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        
        cam_infos.append(RandCameraInfo(
            uid=front_view_count + idx, 
            R=R, T=T, FovY=FovY, FovX=FovX,
            width=image_w, height=image_h, 
            delta_polar=delta_polar[0], 
            delta_azimuth=delta_azimuth[0], 
            delta_radius=delta_radius[0]
        ))
    
    # Shuffle camera order to avoid all front-view cameras in first half
    random.shuffle(cam_infos)
    
    # Re-assign uid for continuity
    for i, cam_info in enumerate(cam_infos):
        cam_infos[i] = cam_info._replace(uid=i)
    
    print(f"[INFO] Final: {front_view_count} front-view cameras and {angle45_view_count} 45-degree cameras, total {size}")
    
    return cam_infos

def GenerateRandomCameras(opt, size=2000, SSAA=True):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses(size, opt, radius_range=opt.radius_range, theta_range=opt.theta_range, phi_range=opt.phi_range, 
                                             angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate,
                                             rand_cam_gamma=opt.rand_cam_gamma)
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
    
    cam_infos = []

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    #generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        # matrix = poses[idx]
        # R = matrix[:3,:3]
        # T = matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=image_w, 
                                        height=image_h, delta_polar = delta_polar[idx],
                                        delta_azimuth = delta_azimuth[idx], delta_radius = delta_radius[idx]))           
    return cam_infos

def GeneratePurnCameras(opt, size=300):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses(size, opt, radius_range=[opt.default_radius,opt.default_radius+0.1], theta_range=opt.theta_range, phi_range=opt.phi_range, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate)
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    #fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
    fov = opt.default_fovy
    cam_infos = []
    #generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])     
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        # matrix = poses[idx]
        # R = matrix[:3,:3]
        # T = matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                        height = opt.image_h, delta_polar = delta_polar[idx],delta_azimuth = delta_azimuth[idx], delta_radius = delta_radius[idx]))           
    return cam_infos

def GenerateRotatingBatchCameras(opt, current_iter, total_iters=5000, views_per_iter=4):
    """
    Generate multiple surrounding camera views per iteration, gradually revolving 360 degrees as iteration count increases
    
    Args:
        opt: configuration options
        current_iter: current iteration count (0-5000)
        total_iters: total iterations to complete one 360-degree revolution, default 5000
        views_per_iter: number of views generated per iteration, default 4
    
    Returns:
        batch_cameras: list of camera info for the current iteration
    """
    # Compute current base angle
    base_phi = (current_iter / total_iters) * 360.0
    
    # Create empty camera info list
    batch_cameras = []
    
    # Generate views_per_iter evenly distributed camera views per iteration
    for i in range(views_per_iter):
        # Compute current azimuth - evenly distributed around base angle
        current_phi = (base_phi + (i * 360.0 / views_per_iter)) % 360.0
        
        # Set camera parameters
        fov = opt.default_fovy
        thetas = torch.FloatTensor([opt.default_polar])  # use default polar angle
        phis = torch.FloatTensor([current_phi])          # current azimuth
        radius = torch.FloatTensor([opt.default_radius]) # use default radius
        
        # Generate camera pose
        poses = circle_poses(
            radius=radius, 
            theta=thetas, 
            phi=phis, 
            angle_overhead=opt.angle_overhead, 
            angle_front=opt.angle_front
        )
        
        # Compute camera matrix
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        
        # Compute field of view
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov
        
        # Compute delta relative to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        
        # Create camera info object
        camera_info = RandCameraInfo(
            uid=current_iter * views_per_iter + i,  # assign a unique ID to each camera
            R=R, 
            T=T, 
            FovY=FovY, 
            FovX=FovX,
            width=opt.image_w, 
            height=opt.image_h, 
            delta_polar=delta_polar,
            delta_azimuth=delta_azimuth, 
            delta_radius=delta_radius
        )
        
        # Add to camera list
        batch_cameras.append(camera_info)
    
    return batch_cameras

def GenerateFullCoverageCameras(opt, current_iter, total_iters=5000, views_per_iter=2, render45=True, num_cycles=10):
    """
    Generate camera views that evenly cover 360 degrees, ensuring all angles are updated multiple times during training
    
    Args:
        opt: configuration options
        current_iter: current iteration count (0-5000)
        total_iters: total iteration count, default 5000
        views_per_iter: number of front-view cameras per iteration, default 2
        render45: whether to generate 45-degree view cameras, default True
        num_cycles: number of full cycles to divide total iterations into, default 10 (i.e., one 360-degree cycle every 500 iterations)
    
    Returns:
        batch_cameras: list of camera info for the current iteration
    """
    cycle_length = total_iters // num_cycles
    current_cycle = current_iter // cycle_length
    cycle_position = (current_iter % cycle_length) / cycle_length
    base_phi = (cycle_position * 360.0 + (current_cycle * (360.0 / num_cycles))) % 360.0

    batch_cameras = []
    
    # Generate front-view cameras
    for i in range(views_per_iter):
        current_phi = (base_phi + (i * 360.0 / views_per_iter)) % 360.0
        thetas = torch.FloatTensor([opt.default_polar])  
        phis = torch.FloatTensor([current_phi])
        radius = torch.FloatTensor([opt.default_radius])

        poses = circle_poses(radius=radius, theta=thetas, phi=phis, 
                             angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        fovy = focal2fov(fov2focal(opt.default_fovy, opt.image_h), opt.image_w)

        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360
        delta_radius = radius - opt.default_radius

        batch_cameras.append(RandCameraInfo(
            uid=current_iter * views_per_iter * (2 if render45 else 1) + i, 
            R=R, T=T, FovY=fovy, FovX=opt.default_fovy,
            width=opt.image_w, height=opt.image_h, 
            delta_polar=delta_polar, delta_azimuth=delta_azimuth, delta_radius=delta_radius
        ))

    # Generate 45-degree view cameras
    if render45:
        print(f"[DEBUG] Generating 45-degree view cameras")
        for i in range(views_per_iter):
            current_phi = (base_phi + (i * 360.0 / views_per_iter)) % 360.0
            thetas = torch.FloatTensor([opt.default_polar * (2 / 3)])  # ensure 45° view
            print(f"[DEBUG] 45-degree camera {i}: theta={thetas.item()}, phi={current_phi}")
            phis = torch.FloatTensor([current_phi])
            radius = torch.FloatTensor([opt.default_radius])

            poses = circle_poses(radius=radius, theta=thetas, phi=phis, 
                                 angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            fovy = focal2fov(fov2focal(opt.default_fovy, opt.image_h), opt.image_w)

            delta_polar = thetas - opt.default_polar
            delta_azimuth = phis - opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360
            delta_radius = radius - opt.default_radius

            batch_cameras.append(RandCameraInfo(
                uid=current_iter * views_per_iter * 2 + views_per_iter + i,  
                R=R, T=T, FovY=fovy, FovX=opt.default_fovy,
                width=opt.image_w, height=opt.image_h, 
                delta_polar=delta_polar, delta_azimuth=delta_azimuth, delta_radius=delta_radius
            ))
    print(f"[DEBUG] Final camera count: {len(batch_cameras)}")
    return batch_cameras


sceneLoadTypeCallbacks = {
    # "Colmap": readColmapSceneInfo,
    # "Blender" : readNerfSyntheticInfo,
    "RandomCam" : readCircleCamInfo
}
