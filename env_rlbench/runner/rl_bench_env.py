# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import sys
sys.path.insert(0, os.getcwd())

import re
import logging
from typing import List, Type
import copy

import numpy as np
from pyrep.const import RenderMode
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects import Dummy, VisionSensor, Shape
from rlbench import ActionMode, ObservationConfig
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from yarr.agents.agent import VideoSummary
from yarr.envs.rlbench_env import RLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition

from env_rlbench_peract.utils.rlbench_utils import MyEnvironmentPeract, get_mask, get_rgb_depth, get_seg_mask
from env_rlbench.runner.rl_bench_camera import get_camera
from env_rlbench_peract.utils.rlbench_objects import task_object_dict

import utils.transform_utils as T
from utils.logger_utils import EnvLogger
from utils.vis_utils import get_vis_pose

RECORD_EVERY = 20


class CustomRLBenchEnv(RLBenchEnv):
    def __init__(
        self,
        task_class: Type[Task],
        observation_config: ObservationConfig,
        action_mode: ActionMode,
        episode_length: int,
        dataset_root: str = "",
        channels_last: bool = False,
        reward_scale=100.0,
        headless: bool = True,
        time_in_state: bool = False,
        enable_stage: bool = False,
        use_fp: bool = False,
        fp_cam_name: str = None,
        pose_estimation_wrapper=None,
    ):
        super(CustomRLBenchEnv, self).__init__(
            task_class,
            observation_config,
            action_mode,
            dataset_root,
            channels_last,
            headless=headless,
        )
        self._reward_scale = reward_scale
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._i = 0

        self._last_mile = False
        self._start_tracking = False

        # support object state observation
        self._rlbench_env = MyEnvironmentPeract(
            action_mode=action_mode,
            obs_config=observation_config,
            dataset_root=dataset_root,
            headless=headless,
        )
        task_name = re.sub(
            '(?<!^)(?=[A-Z])', '_', task_class.__name__).lower()
        self.grasp_obj_name = None
        self.target_obj_name = None
        self.enable_stage = enable_stage
        if self.enable_stage:
            self._cur_stage = 0

        # ------------------------
        # |    Pose Estimation   |
        # ------------------------
        self.use_fp = use_fp
        self.fp_cam_name = fp_cam_name
        self.pose_estimation_wrapper = pose_estimation_wrapper
        self.pose_estimator = None
        self.current_pose_estimation = None

        # ------------------------
        # |        Logging       |
        # ------------------------
        self.env_logger = EnvLogger()
        self.env_logger.add_data_type("obs_record_rgb", "rgb")
        self.env_logger.add_data_type("obs_wrist_rgb", "rgb")

        self.env_logger.add_data_type("obs_fp_rgb", "rgb")
        self.env_logger.add_data_type("obs_fp_depth", "depth")
        self.env_logger.add_data_type("obs_fp_mask", "mask")

        self.env_logger.add_data_type("vis_pose", "rgb")        
        self.env_logger.add_data_type("pose_est", "list")
        self.env_logger.add_data_type("pose_gt", "list")


        self.debug = False
        self._count = 0 # debug only

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if oe.name == "low_dim_state":
                oe.shape = (
                    oe.shape[0] - 7 * 2 + int(self._time_in_state),
                )  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        obs_elems.append(ObservationElement("gripper_pose", (7,), np.float32))
        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0.0, 0.04
            )

        obs_dict = super(CustomRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (
                1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))
            ) * 2.0 - 1.0
            obs_dict["low_dim_state"] = np.concatenate(
                [obs_dict["low_dim_state"], [time]]
            ).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos

        obs_dict["gripper_pose"] = grip_pose
        return obs_dict

    def launch(self):
        super(CustomRLBenchEnv, self).launch()
        self._rlbench_env._scene.enable_stage = self.enable_stage
        self._task._scene.register_step_callback(self._my_callback)

        self._record_cam = VisionSensor.create([512, 288])
        self._record_cam.set_explicit_handling(True)
        pose = np.array(VisionSensor('cam_front').get_pose())
        self._record_cam.set_pose(list(pose))
        self._record_cam.set_render_mode(RenderMode.OPENGL)

        self._wrist_cam = VisionSensor('cam_wrist')
        self._wrist_cam.set_resolution([288, 288])
        self._wrist_cam.set_explicit_handling(True)
        self._wrist_cam.set_render_mode(RenderMode.OPENGL)

        self._overhead_cam = VisionSensor('cam_overhead')
        self._overhead_cam.set_resolution([288, 288])
        self._overhead_cam.set_explicit_handling(True)
        self._overhead_cam.set_render_mode(RenderMode.OPENGL)

        # get fp camera
        if self.use_fp and self.fp_cam_name is not None:
            if self.fp_cam_name == "cam_front":
                self._fp_cam = VisionSensor.create([512, 288])
                self._fp_cam.set_explicit_handling(True)
                pose = np.array(VisionSensor('cam_front').get_pose())
                self._fp_cam.set_pose(list(pose))
                self._fp_cam.set_render_mode(RenderMode.OPENGL)

                self._fp_cam_mask = VisionSensor.create([512, 288])
                self._fp_cam_mask.set_explicit_handling(True)
                pose = np.array(VisionSensor('cam_front_mask').get_pose())
                self._fp_cam_mask.set_pose(list(pose))
                self._fp_cam_mask.set_render_mode(RenderMode.OPENGL_COLOR_CODED)
            elif self.fp_cam_name == "cam_wrist":
                self._fp_cam = copy.deepcopy(VisionSensor('cam_wrist'))
                self._fp_cam.set_resolution([288, 288])
                self._fp_cam.set_explicit_handling(True)

                self._fp_cam_mask = copy.deepcopy(VisionSensor('cam_wrist_mask'))
                self._fp_cam_mask.set_resolution([288, 288])
                self._fp_cam_mask.set_explicit_handling(True)
            elif self.fp_cam_name == "cam_overhead":
                self._fp_cam = copy.deepcopy(VisionSensor('cam_overhead'))
                self._fp_cam.set_resolution([288, 288])
                self._fp_cam.set_explicit_handling(True)

                self._fp_cam_mask = copy.deepcopy(VisionSensor('cam_overhead_mask'))
                self._fp_cam_mask.set_resolution([288, 288])
                self._fp_cam_mask.set_explicit_handling(True)
            else:
                self._fp_cam = get_camera(self.fp_cam_name)

                self._fp_cam_mask = VisionSensor.create([1024, 576])
                self._fp_cam_mask.set_explicit_handling(True)
                pose = np.array(self._fp_cam.get_pose())
                self._fp_cam_mask.set_pose(list(pose))
                self._fp_cam_mask.set_render_mode(RenderMode.OPENGL_COLOR_CODED)
    
        # if self.debug:
        #     pose = np.array(self._fp_cam.get_pose())
        #     orientation = np.array(self._fp_cam.get_orientation())
        #     print(pose, orientation)

        #     pose = np.array(VisionSensor('cam_front').get_pose())
        #     orientation = np.array(VisionSensor('cam_front').get_orientation())
        #     print(pose, orientation)
        #     pose = np.array(VisionSensor('cam_over_shoulder_right_mask').get_pose())
        #     orientation = np.array(VisionSensor('cam_over_shoulder_right_mask').get_orientation())
        #     print(pose, orientation)

    def reset(self) -> dict:
        self._previous_obs_dict = super(CustomRLBenchEnv, self).reset()

        self._i = 0
        self._cur_stage = 0

        self._last_mile = False
        self._start_tracking = False

        self.env_logger.clear()
        
        # get pose estimator
        if self.pose_estimation_wrapper is not None:
            self.end_tracking()
        
        return self._previous_obs_dict
    
    def reset_to_demo(self, demo):
        self._task.reset_to_demo(demo)  # TaskEnvironment().reset_to_demo()

        self._i = 0
        self._cur_stage = 0

        self._last_mile = False
        self._start_tracking = False
        
        self.env_logger.clear()
        
        # get pose estimator
        if self.pose_estimation_wrapper is not None:
            self.end_tracking()
    
    def set_variation(self, variation_number):
        self._task.set_variation(variation_number)

        task_name = self._task.get_name()
        if self.enable_stage:
            self.grasp_obj_name = task_object_dict[task_name]["grasp_object_name"] if isinstance(task_object_dict[task_name]["grasp_object_name"], str) else task_object_dict[task_name]["grasp_object_name"][self._cur_stage]
            self.grasp_obj_name = self.grasp_obj_name[variation_number] if isinstance(self.grasp_obj_name, dict) else self.grasp_obj_name
            self.target_obj_name = task_object_dict[task_name]["target_object_name"] if isinstance(task_object_dict[task_name]["target_object_name"], str) else task_object_dict[task_name]["target_object_name"][self._cur_stage]
            self.target_obj_name = self.target_obj_name[variation_number] if isinstance(self.target_obj_name, dict) else self.target_obj_name
        else:
            self.grasp_obj_name = task_object_dict[task_name]["grasp_object_name"] if isinstance(task_object_dict[task_name]["grasp_object_name"], str) else task_object_dict[task_name]["grasp_object_name"][variation_number]
            self.target_obj_name = task_object_dict[task_name]["target_object_name"] if isinstance(task_object_dict[task_name]["target_object_name"], str) else task_object_dict[task_name]["target_object_name"][variation_number]
            if task_name == "insert_onto_square_peg":
                self.target_obj_name = self._task._task._chosen_pillar_name

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _get_pose_est(self):
        def fp_frame_to_world_frame(fp_mat, R):
            # flip x- and y-axis
            # tf_flip_x_y = np.eye(4)
            # tf_flip_x_y[0, 0] = -1   # flip x-axis
            # tf_flip_x_y[1, 1] = -1   # flip y-axis

            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler('zyx', [np.pi, 0, 0])
            tf_flip_x_y = np.eye(4)
            tf_flip_x_y[:3, :3] = r.as_matrix()

            # to world frame
            tf_camera_to_world = R

            fp_mat_flipped = np.matmul(tf_flip_x_y, fp_mat)              # flip x- and y-axis
            world_mat = np.matmul(tf_camera_to_world, fp_mat_flipped)    # to world frame
            pose = np.concatenate([world_mat[:3, 3], T.mat2quat(world_mat[:3, :3])])
            return pose, world_mat
        
        def world_frame_to_fp_frame(world_pose, R):
            # flip x- and y-axis
            # tf_flip_x_y = np.eye(4)
            # tf_flip_x_y[0, 0] = -1   # flip x-axis
            # tf_flip_x_y[1, 1] = -1   # flip y-axis

            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler('zyx', [np.pi, 0, 0])
            tf_flip_x_y = np.eye(4)
            tf_flip_x_y[:3, :3] = r.as_matrix()

            # to world frame
            tf_camera_to_world = R

            world_mat = np.eye(4)
            world_mat[:3, 3] = world_pose[:3]
            world_mat[:3, :3] = T.quat2mat(world_pose[3:])
            fp_mat_flipped = np.matmul(np.linalg.inv(tf_camera_to_world), world_mat)    # to world frame
            fp_mat = np.matmul(np.linalg.inv(tf_flip_x_y), fp_mat_flipped)
            return fp_mat
        

        # get rgb depth
        fp_rgb, fp_depth, fp_pcd = get_rgb_depth(
            sensor=self._fp_cam, 
            get_rgb=True, get_depth=True, get_pcd=True, 
            rgb_noise=None, depth_noise=None, 
            depth_in_meters=True,
        )

        # get mask
        mask = get_mask(sensor=self._fp_cam_mask, masks_as_one_channel=True)
        fp_mask = get_seg_mask([Shape(self.grasp_obj_name)], mask)
            
        # get new pose estimation
        K = self._fp_cam.get_intrinsic_matrix()
        R = self._fp_cam.get_matrix()
        K[:2, :2] *= -1 # !! convert into positive
        color = fp_rgb
        depth = fp_depth
        select_seg_id = 1 # !! 0 is background
        ob_mask = (fp_mask == select_seg_id).astype(bool).astype(np.uint8)

        if self.current_pose_estimation is None or self._i % 5 == 0:
            fp_mat = self.pose_estimator.register(K=K, rgb=color, depth=depth, ob_mask=ob_mask, iteration=5)
            self._count += 1
        else:
            if self.pose_estimator.pose_last is None:
                fp_mat = self.pose_estimator.register(K=K, rgb=color, depth=depth, ob_mask=ob_mask, iteration=5)
            else:
                fp_mat = self.pose_estimator.track_one(rgb=color, depth=depth, K=K, iteration=2)
            self._count += 1

        # convert pose estimation to world frame
        pose_est, world_mat = fp_frame_to_world_frame(fp_mat, R)
        
        if self.debug:
            from scipy.spatial.transform import Rotation
            def get_euler(pose):
                rot = Rotation.from_quat(pose[3:]) # (x, y, z, w)
                euler = rot.as_euler('xyz', degrees=True)
                return euler
            def get_rotvec(pose):
                rot = Rotation.from_quat(pose[3:]) # (x, y, z, w)
                rotvec = rot.as_rotvec('xyz')
                return rotvec
            
            from foundation_pose.Utils import depth2xyzmap, toOpen3dCloud
            import open3d as o3d

            # save intrinsics
            debug_dir = "./debug"
            np.savetxt(os.path.join(debug_dir, "K.txt"), K)

            # save rgb
            import cv2
            cv2.imwrite(os.path.join(debug_dir, "fp_rgb.png"), cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
            
            # save depth
            cv2.imwrite(os.path.join(debug_dir, "fp_depth.png"), (fp_depth * 1000).astype(np.uint16))
            print(np.max(depth), np.min(depth))
            # load depth
            depth_map = cv2.imread(os.path.join(debug_dir, "fp_depth.png"), cv2.IMREAD_ANYDEPTH) / 1000.
            print(np.max(depth_map), np.min(depth_map))

            # save mask
            cv2.imwrite(os.path.join(debug_dir, "fp_mask.png"), (ob_mask*255.0).clip(0,255))

            # save pointcloud in simulation's global frame (ground truth)
            pcd = toOpen3dCloud(fp_pcd.reshape(-1, 3), fp_rgb.reshape(-1, 3))
            o3d.io.write_point_cloud(f'{debug_dir}/fp_pcd.ply', pcd)    # global frame

            # save mesh in simulation's global frame (converted from FP) - #*the mesh should align with fp_pcd
            m = self.pose_estimator.mesh.copy()
            m.apply_transform(world_mat)
            m.export(f'{debug_dir}/model_tf_world_fp.obj')
        
            # save pointcloud in camera frame (directly from FP)
            xyz_map = depth2xyzmap(depth, K)
            valid = depth>=0.001
            pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)        # camera frame

            # save mesh in camera frame (directly from FP) - #*the mesh should align with xyz_map
            m = self.pose_estimator.mesh.copy()
            m.apply_transform(fp_mat)
            m.export(f'{debug_dir}/model_tf.obj')

            # check determinant (1 if in right-hand coordinate)
            print("fp_mat det", np.linalg.det(fp_mat[:3, :3]))
            print("world_mat det", np.linalg.det(world_mat[:3, :3]))

            pose_gt = Shape(self.grasp_obj_name).get_pose()
            print("pose_est", pose_est)
            print("pose_gt", pose_gt)
            print("pose_est_euler", get_euler(pose_est))
            print("pose_gt_euler", get_euler(pose_gt))

            # save mesh in simulation's global frame (converted from FP) - #*the mesh should align with fp_pcd
            world_mat = np.eye(4)
            world_mat[:3, 3] = pose_est[:3]
            world_mat[:3, :3] = Rotation.from_quat(pose_est[3:]).as_matrix()
            m = self.pose_estimator.mesh.copy()
            m.apply_transform(world_mat)
            m.export(f'{debug_dir}/model_tf_world_fp.obj')

            # save mesh in simulation's global frame (ground truth) - #*the mesh should align with fp_pcd
            world_mat = np.eye(4)
            world_mat[:3, 3] = pose_gt[:3]
            world_mat[:3, :3] = Rotation.from_quat(pose_gt[3:]).as_matrix()
            m = self.pose_estimator.mesh.copy()
            m.apply_transform(world_mat)
            m.export(f'{debug_dir}/model_tf_world_gt.obj')

            exit()

        # logging
        self.env_logger.add_data("obs_fp_rgb", fp_rgb)
        self.env_logger.add_data("obs_fp_depth", fp_depth)
        self.env_logger.add_data("obs_fp_mask", fp_mask)

        vis_pose = get_vis_pose(
            pose=fp_mat, 
            color=color, 
            K=K, 
            mesh=self.pose_estimation_wrapper.mesh
        )

        pose_gt = Shape(self.grasp_obj_name).get_pose()
        self.env_logger.add_data("pose_est", pose_est)
        self.env_logger.add_data("pose_gt", pose_gt)
        
        return pose_est, vis_pose

    def _my_callback(self):

        # last mile
        grasp_obj_pose = Shape(self.grasp_obj_name).get_pose()
        target_obj_pose = Shape(self.target_obj_name).get_pose()
        if np.linalg.norm(grasp_obj_pose[:3] - target_obj_pose[:3]) < 0.15:
            self._last_mile = True

        # fp pose
        if self.use_fp:
            if self._start_tracking and (self.current_pose_estimation is None or self._i % 1 == 0):
                self.current_pose_estimation, vis_pose = self._get_pose_est()
                cap_wrist = (self._wrist_cam.capture_rgb() * 255).astype(np.uint8)
                # self.env_logger.add_data("vis_pose", np.concatenate([vis_pose, cap_wrist], axis=1))
                self.env_logger.add_data("vis_pose", vis_pose)
            else:
                self._fp_cam.handle_explicitly()
                cap_fp = (self._fp_cam.capture_rgb() * 255).astype(np.uint8)
                cap_wrist = (self._wrist_cam.capture_rgb() * 255).astype(np.uint8)
                # self.env_logger.add_data("vis_pose", np.concatenate([cap_fp, cap_wrist], axis=1))
                self.env_logger.add_data("vis_pose", cap_fp)

        # record cam (vis only)
        self._record_cam.handle_explicitly()
        self._overhead_cam.handle_explicitly()
        cap_record = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        cap_overhead = (self._overhead_cam.capture_rgb() * 255).astype(np.uint8)
        self.env_logger.add_data("obs_record_rgb", np.concatenate([cap_record, cap_overhead], axis=1))

        # wrist cam (vis only)
        self._wrist_cam.handle_explicitly()
        cap_wrist = (self._wrist_cam.capture_rgb() * 255).astype(np.uint8)
        self.env_logger.add_data("obs_wrist_rgb", cap_wrist)

    def step(self, action: np.ndarray, record: bool = False, verbose: bool = False) -> Transition:
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        ik_error = False
        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            if verbose:
                print(e)
            terminal = True
            reward = 0.0
            ik_error = True

        self._i += 1

        return Transition(obs, reward, terminal)

    def run_demo_until_n_waypoint(self, start=0, n_waypoint=-1):
        from pyrep.const import ObjectType
        from pyrep.errors import ConfigurationPathError
        from rlbench.backend.exceptions import (
            WaypointError, BoundaryError, NoWaypointsError, DemoError)
        
        randomly_place = False
        scene = self.env._scene
        if not scene._has_init_task:
            scene.init_task()
        if not scene._has_init_episode:
            scene.init_episode(scene._variation_index,
                                randomly_place=randomly_place)
        # scene._has_init_episode = False

        waypoints = self.env._scene.task.get_waypoints()
        if n_waypoint > 0:
            waypoints = waypoints[start:n_waypoint]

        for i, point in enumerate(waypoints):
            point.start_of_path()
            if point.skip:
                continue
            grasped_objects = scene.robot.gripper.get_grasped_objects()
            colliding_shapes = [s for s in scene.pyrep.get_objects_in_tree(
                object_type=ObjectType.SHAPE) if s not in grasped_objects
                                and s not in scene._robot_shapes and s.is_collidable()
                                and scene.robot.arm.check_arm_collision(s)]
            [s.set_collidable(False) for s in colliding_shapes]
            try:
                path = point.get_path()
                [s.set_collidable(True) for s in colliding_shapes]
            except ConfigurationPathError as e:
                [s.set_collidable(True) for s in colliding_shapes]
                raise DemoError(
                    'Could not get a path for waypoint %d.' % i,
                    scene.task) from e
            ext = point.get_ext()
            path.visualize()

            done = False
            success = False
            while not done:
                done = path.step()
                scene.step()
                # self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                # self._demo_record_step(demo, record, callable_each_step)
                success, term = scene.task.success()

            point.end_of_path()

            path.clear_visualization()
            if len(ext) > 0:
                contains_param = False
                start_of_bracket = -1
                gripper = scene.robot.gripper
                if 'open_gripper(' in ext:
                    gripper.release()
                    start_of_bracket = ext.index('open_gripper(') + 13
                    contains_param = ext[start_of_bracket] != ')'
                    if not contains_param:
                        done = False
                        while not done:
                            gripper_open = 1.0
                            done = gripper.actuate(gripper_open, 0.04)
                            scene.step()
                            # self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                            # if self._obs_config.record_gripper_closing:
                            #     self._demo_record_step(
                            #         demo, record, callable_each_step)
                elif 'close_gripper(' in ext:
                    start_of_bracket = ext.index('close_gripper(') + 14
                    contains_param = ext[start_of_bracket] != ')'
                    if not contains_param:
                        done = False
                        while not done:
                            gripper_open = 0.0
                            done = gripper.actuate(gripper_open, 0.04)
                            scene.step()
                            # self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                            # if self._obs_config.record_gripper_closing:
                            #     self._demo_record_step(
                            #         demo, record, callable_each_step)

                if contains_param:
                    rest = ext[start_of_bracket:]
                    num = float(rest[:rest.index(')')])
                    done = False
                    while not done:
                        gripper_open = num
                        done = gripper.actuate(gripper_open, 0.04)
                        scene.step()
                        # self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                        # if self._obs_config.record_gripper_closing:
                        #     self._demo_record_step(
                        #         demo, record, callable_each_step)

                if 'close_gripper(' in ext:
                    for g_obj in scene.task.get_graspable_objects():
                        gripper.grasp(g_obj)

    def start_tracking(self):
        assert self.pose_estimation_wrapper is not None

        if self.use_fp:
            # obj_name_without_digit = ''.join(c for c in self.grasp_obj_name if not c.isdigit())  # no need to remove digit
            self.pose_estimation_wrapper.update_grasp_obj_name(self.grasp_obj_name) 
            
            print("create FP estimator...", self.grasp_obj_name)
            self.pose_estimator = self.pose_estimation_wrapper.create_estimator(debug_level=0)

            from Utils import trimesh_add_pure_colored_texture
            task_name = self._task.get_name()
            if task_name == "stack_cups" or task_name == "stack_blocks":
                current_rgb = self._task._task.select_colors[self._cur_stage]
                current_rgb = np.array(current_rgb)
                mesh = self.pose_estimation_wrapper.mesh
                mesh = trimesh_add_pure_colored_texture(mesh, color=current_rgb*255)
                mesh.visual.vertex_colors = np.tile((current_rgb*255).reshape(1,3), (len(mesh.vertices), 1))

            self.current_pose_estimation = None
            self._start_tracking = True

    def end_tracking(self):
        assert self.pose_estimation_wrapper is not None
        self.pose_estimator = None
        self.current_pose_estimation = None
        self._start_tracking = False

    def move_to_next_stage(self):
        self._cur_stage += 1
        task_name = self._task.get_name()
        variation_number = self._task._scene._variation_index
        if self.enable_stage:
            self.grasp_obj_name = task_object_dict[task_name]["grasp_object_name"] if isinstance(task_object_dict[task_name]["grasp_object_name"], str) else task_object_dict[task_name]["grasp_object_name"][self._cur_stage]
            self.grasp_obj_name = self.grasp_obj_name[variation_number] if isinstance(self.grasp_obj_name, dict) else self.grasp_obj_name
            self.target_obj_name = task_object_dict[task_name]["target_object_name"] if isinstance(task_object_dict[task_name]["target_object_name"], str) else task_object_dict[task_name]["target_object_name"][self._cur_stage]
            self.target_obj_name = self.target_obj_name[variation_number] if isinstance(self.target_obj_name, dict) else self.target_obj_name
        self.pose_estimator = None
        self.current_pose_estimation = None
        self._start_tracking = False

    def get_env_logger(self):
        return self.env_logger
        