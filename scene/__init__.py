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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks,GenerateRandomCameras,GeneratePurnCameras,GenerateCircleCameras, GenerateRandomFixedAngleCameras
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, GenerateCamParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_RcamInfos

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, pose_args : GenerateCamParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args._model_path
        self.pretrained_model_path = args.pretrained_model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.resolution_scales = resolution_scales
        self.pose_args = pose_args
        self.args = args
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.test_cameras = {}
        scene_info = sceneLoadTypeCallbacks["RandomCam"](self.model_path ,pose_args)

        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = pose_args.default_radius #    scene_info.nerf_normalization["radius"]
        for resolution_scale in resolution_scales:
            self.test_cameras[resolution_scale] = cameraList_from_RcamInfos(scene_info.test_cameras, resolution_scale, self.pose_args)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        elif self.pretrained_model_path is not None:
            self.gaussians.load_ply(self.pretrained_model_path)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getRandTrainCameras(self, scale=1.0):
        rand_train_cameras = GenerateRandomCameras(self.pose_args, self.args.batch, SSAA=True)
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(rand_train_cameras, resolution_scale, self.pose_args, SSAA=True)        
        return train_cameras[scale]
    
    # def getRandomFixedAngleCameras(self, scale=1.0, angle_ratio=0.5):
    #     """
    #     Get random fixed-angle cameras (front and 45-degree views only)
        
    #     Args:
    #         scale: resolution scale, default 1.0
    #         angle_ratio: ratio of front vs 45-degree views, default 0.5 means equal split
        
    #     Returns:
    #         list of cameras at random fixed angles
    #     """
    #     # Generate random fixed-angle cameras
    #     fixed_angle_cameras = GenerateRandomFixedAngleCameras(
    #         self.pose_args, 
    #         self.args.batch, 
    #         SSAA=True, 
    #         angle_ratio=angle_ratio
    #     )
        
    #     # Create camera list
    #     train_cameras = {}
    #     for resolution_scale in self.resolution_scales:
    #         train_cameras[resolution_scale] = cameraList_from_RcamInfos(
    #             fixed_angle_cameras, 
    #             resolution_scale, 
    #             self.pose_args, 
    #             SSAA=True
    #         )
        
    #     return train_cameras[scale]

    def getRandomFixedAngleCameras(self, scale=1.0, random_ratio=True, min_front_ratio=0.3, max_front_ratio=0.7):
        """
        Get random fixed-angle cameras (with randomly assigned proportions)
        """
        # Generate random fixed-angle cameras
        fixed_angle_cameras = GenerateRandomFixedAngleCameras(
            self.pose_args, 
            self.args.batch, 
            SSAA=True, 
            random_ratio=random_ratio,
            min_front_ratio=min_front_ratio,
            max_front_ratio=max_front_ratio
        )
        
        # Create camera list
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(
                fixed_angle_cameras, 
                resolution_scale, 
                self.pose_args, 
                SSAA=True
            )
        
        return train_cameras[scale]

    def getPurnTrainCameras(self, scale=1.0):
        rand_train_cameras = GeneratePurnCameras(self.pose_args)
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(rand_train_cameras, resolution_scale, self.pose_args)        
        return train_cameras[scale]


    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getCircleVideoCameras(self, scale=1.0, batch_size=120, render45 = True):
        video_circle_cameras = GenerateCircleCameras(self.pose_args,batch_size,render45)
        video_cameras = {}
        for resolution_scale in self.resolution_scales:
            video_cameras[resolution_scale] = cameraList_from_RcamInfos(video_circle_cameras, resolution_scale, self.pose_args)        
        return video_cameras[scale]
    
    def getRotatingBatchCameras(self, current_iter, total_iters=5000, views_per_iter=4, scale=1.0):
        """
        Get multiple surrounding camera views for each iteration
        
        Args:
            current_iter: current iteration count
            total_iters: total iterations needed to complete one 360-degree rotation, default 5000
            views_per_iter: number of views generated per iteration, default 4
            scale: resolution scale, default 1.0
        
        Returns:
            list of cameras for the current iteration
        """
        from scene.dataset_readers import GenerateRotatingBatchCameras
        
        # Generate multiple surrounding camera views for the current iteration
        rotating_cameras = GenerateRotatingBatchCameras(
            self.pose_args, 
            current_iter, 
            total_iters, 
            views_per_iter
        )
        
        # Create camera list
        rotating_cams = {}
        for resolution_scale in self.resolution_scales:
            rotating_cams[resolution_scale] = cameraList_from_RcamInfos(
                rotating_cameras, 
                resolution_scale, 
                self.pose_args
            )
        
        return rotating_cams[scale]
    
    def getFullCoverageCameras(self, current_iter, total_iters=5000, views_per_iter=2, scale=1.0, render45=True, num_cycles=10):
        """
        Get camera views that evenly cover the full 360 degrees, ensuring all angles are updated multiple times during training
        
        Args:
            current_iter: current iteration count
            total_iters: total iteration count, default 5000
            views_per_iter: number of front-view cameras per iteration, default 2
            scale: resolution scale, default 1.0
            render45: whether to generate 45-degree view cameras, default True
            num_cycles: number of cycles to divide total iterations into, default 10
        
        Returns:
            list of cameras for the current iteration
        """
        from scene.dataset_readers import GenerateFullCoverageCameras
        
        # Generate cameras covering all viewing angles
        full_coverage_cameras = GenerateFullCoverageCameras(
            self.pose_args, 
            current_iter, 
            total_iters, 
            views_per_iter,
            render45,
            num_cycles
        )
        
        # Create camera list
        coverage_cams = {}
        for resolution_scale in self.resolution_scales:
            coverage_cams[resolution_scale] = cameraList_from_RcamInfos(
                full_coverage_cameras, 
                resolution_scale, 
                self.pose_args
            )
        
        return coverage_cams[scale]