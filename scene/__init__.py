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
    #     获取随机固定角度相机（只包含正面和45度视角）
        
    #     Args:
    #         scale: 分辨率缩放比例，默认为1.0
    #         angle_ratio: 正面视角和45度视角的比例，默认0.5表示各占一半
        
    #     Returns:
    #         随机固定角度的相机列表
    #     """
    #     # 生成随机固定角度相机
    #     fixed_angle_cameras = GenerateRandomFixedAngleCameras(
    #         self.pose_args, 
    #         self.args.batch, 
    #         SSAA=True, 
    #         angle_ratio=angle_ratio
    #     )
        
    #     # 创建相机列表
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
        获取随机固定角度相机（随机比例分配）
        """
        # 生成随机固定角度相机
        fixed_angle_cameras = GenerateRandomFixedAngleCameras(
            self.pose_args, 
            self.args.batch, 
            SSAA=True, 
            random_ratio=random_ratio,
            min_front_ratio=min_front_ratio,
            max_front_ratio=max_front_ratio
        )
        
        # 创建相机列表
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
        获取每次迭代的多个环绕相机视角
        
        Args:
            current_iter: 当前迭代次数
            total_iters: 完成一个360度旋转所需的总迭代次数，默认为5000
            views_per_iter: 每次迭代生成的视角数量，默认为4
            scale: 分辨率缩放比例，默认为1.0
        
        Returns:
            当前迭代的多个相机列表
        """
        from scene.dataset_readers import GenerateRotatingBatchCameras
        
        # 生成当前迭代的多个环绕相机视角
        rotating_cameras = GenerateRotatingBatchCameras(
            self.pose_args, 
            current_iter, 
            total_iters, 
            views_per_iter
        )
        
        # 创建相机列表
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
        获取均匀覆盖全部360度的相机视角，确保所有视角都能在整个训练过程中多次更新
        
        Args:
            current_iter: 当前迭代次数
            total_iters: 总迭代次数，默认为5000
            views_per_iter: 每次迭代生成的正面视角数量，默认为2
            scale: 分辨率缩放比例，默认为1.0
            render45: 是否生成45度视角相机，默认为True
            num_cycles: 将5000次迭代分成多少个周期，默认为10
        
        Returns:
            当前迭代的多个相机列表
        """
        from scene.dataset_readers import GenerateFullCoverageCameras
        
        # 生成覆盖所有视角的相机
        full_coverage_cameras = GenerateFullCoverageCameras(
            self.pose_args, 
            current_iter, 
            total_iters, 
            views_per_iter,
            render45,
            num_cycles
        )
        
        # 创建相机列表
        coverage_cams = {}
        for resolution_scale in self.resolution_scales:
            coverage_cams[resolution_scale] = cameraList_from_RcamInfos(
                full_coverage_cameras, 
                resolution_scale, 
                self.pose_args
            )
        
        return coverage_cams[scale]