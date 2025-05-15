# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""
Wrappers for collecting object poses from demos
"""
from functools import partial
from os.path import exists, dirname, abspath, join
from typing import List, Callable

import numpy as np
from pyrep import PyRep
from pyrep.objects.object import Object, object_type_to_class
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.backend.const import *
from rlbench.const import SUPPORTED_ROBOTS
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.environment import Environment, DIR_PATH

from env_rlbench_peract.utils.rlbench_objects import task_object_dict


class MyScene(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_stage = False
    
    def _get_misc(self):
        def _get_cam_data(cam: VisionSensor, name: str):
            d = {}
            if cam.still_exists():
                d = {
                    '%s_extrinsics' % name: cam.get_matrix(),
                    '%s_intrinsics' % name: cam.get_intrinsic_matrix(),
                    '%s_near' % name: cam.get_near_clipping_plane(),
                    '%s_far' % name: cam.get_far_clipping_plane(),
                }
            return d
        misc = _get_cam_data(self._cam_over_shoulder_left, 'left_shoulder_camera')
        misc.update(_get_cam_data(self._cam_over_shoulder_right, 'right_shoulder_camera'))
        misc.update(_get_cam_data(self._cam_overhead, 'overhead_camera'))
        misc.update(_get_cam_data(self._cam_front, 'front_camera'))
        misc.update(_get_cam_data(self._cam_wrist, 'wrist_camera'))
        misc.update({"variation_index": self._variation_index})
        
        # * olde version does not have _joint_position_action attribute
        # if self._joint_position_action is not None:
        #     # Store the actual requested joint positions during demo collection
        #     misc.update({"joint_position_action": self._joint_position_action})
        # joint_poses = [j.get_pose() for j in self.robot.arm.joints]
        # misc.update({'joint_poses': joint_poses})

        ee_pose = self.robot.arm.get_tip().get_pose()
        misc.update({f'ee_pose': ee_pose})

        # for multi stage task (specified object manually)
        if self.enable_stage:

            task_name = self.task.get_name()
            task_variation_number = self._variation_index
            stage_num = len(task_object_dict[task_name]["grasp_object_name"]) if isinstance(task_object_dict[task_name]["grasp_object_name"], dict) else len(task_object_dict[task_name]["target_object_name"])
            
            for stage_idx in range(stage_num):
                # add object pose
                grasp_obj_name = task_object_dict[task_name]["grasp_object_name"] if isinstance(task_object_dict[task_name]["grasp_object_name"], str) else task_object_dict[task_name]["grasp_object_name"][stage_idx]
                grasp_obj_name = grasp_obj_name[task_variation_number] if isinstance(grasp_obj_name, dict) else grasp_obj_name
                misc.update({f'stage{stage_idx}_grasp_obj_name': grasp_obj_name})
                grasp_obj_pose = Shape(grasp_obj_name).get_pose() #self.task._cups[0].get_pose()
                misc.update({f'stage{stage_idx}_grasp_obj_pose': grasp_obj_pose})

                target_obj_name = task_object_dict[task_name]["target_object_name"] if isinstance(task_object_dict[task_name]["target_object_name"], str) else task_object_dict[task_name]["target_object_name"][stage_idx]
                target_obj_name = target_obj_name[task_variation_number] if isinstance(target_obj_name, dict) else target_obj_name
                misc.update({f'stage{stage_idx}_target_obj_name': target_obj_name})
                target_obj_pose = Shape(target_obj_name).get_pose()
                misc.update({f'stage{stage_idx}_target_obj_pose': target_obj_pose})


                from utils.pose_utils import get_rel_pose, euler_from_quaternion
                obj_pose_relative_to_target = get_rel_pose(target_obj_pose, grasp_obj_pose) # !! be careful about the order (target, grasp_object)
                misc.update({f'stage{stage_idx}_obj_pose_relative_to_target': obj_pose_relative_to_target})
                target_pose_relative_to_target = get_rel_pose(target_obj_pose, target_obj_pose)
                misc.update({f'stage{stage_idx}_target_pose_relative_to_target': target_pose_relative_to_target})

                # convert from quaternion to euler
                obj_pose_relative_to_target_euler = np.zeros(6)
                obj_pose_relative_to_target_euler[:3] = obj_pose_relative_to_target[:3]
                obj_pose_relative_to_target_euler[3:] = euler_from_quaternion(*obj_pose_relative_to_target[3:])
                misc.update({f'stage{stage_idx}_obj_pose_relative_to_target_euler': obj_pose_relative_to_target_euler})
                target_pose_relative_to_target_euler = np.zeros(6)
                target_pose_relative_to_target_euler[3:] = target_pose_relative_to_target[:3]
                target_pose_relative_to_target_euler[3:] = euler_from_quaternion(*target_pose_relative_to_target[3:])
                misc.update({f'stage{stage_idx}_target_pose_relative_to_target_euler': target_pose_relative_to_target_euler})

        else:
            # add object pose
            task_name = self.task.get_name()
            task_variation_number = self._variation_index
            grasp_obj_name = task_object_dict[task_name]["grasp_object_name"] if isinstance(task_object_dict[task_name]["grasp_object_name"], str) else task_object_dict[task_name]["grasp_object_name"][task_variation_number]
            misc.update({f'grasp_obj_name': grasp_obj_name})
            grasp_obj_pose = Shape(grasp_obj_name).get_pose() #self.task._cups[0].get_pose()
            misc.update({'grasp_obj_pose': grasp_obj_pose})

            target_obj_name = task_object_dict[task_name]["target_object_name"] if isinstance(task_object_dict[task_name]["target_object_name"], str) else task_object_dict[task_name]["target_object_name"][task_variation_number]
            misc.update({f'target_obj_name': target_obj_name})
            if task_name == "insert_onto_square_peg":
                target_obj_name = self.task._chosen_pillar_name
            target_obj_pose = Shape(target_obj_name).get_pose()
            misc.update({'target_obj_pose': target_obj_pose})


            from utils.pose_utils import get_rel_pose, euler_from_quaternion
            # obj_pose_relative_to_init = get_rel_pose_init(init_pose, grasped_obj)
            obj_pose_relative_to_target = get_rel_pose(target_obj_pose, grasp_obj_pose) # !! be careful about the order (target, grasp_object)
            misc.update({'obj_pose_relative_to_target': obj_pose_relative_to_target})
            # cab_pose = get_pose(cabinet)
            # cab_pose_relative_to_obj = get_rel_pose(grasped_obj, cabinet)
            target_pose_relative_to_target = get_rel_pose(target_obj_pose, target_obj_pose)
            misc.update({'target_pose_relative_to_target': target_pose_relative_to_target})

            # (debug only) convert back to world frame
            # print(realtive_to_cab_to_world(env, obj_pose_relative_to_cab, cabinet), obj_pose) # should be the same
            # exit()

            # convert from quaternion to euler
            obj_pose_relative_to_target_euler = np.zeros(6)
            obj_pose_relative_to_target_euler[:3] = obj_pose_relative_to_target[:3]
            obj_pose_relative_to_target_euler[3:] = euler_from_quaternion(*obj_pose_relative_to_target[3:])
            misc.update({'obj_pose_relative_to_target_euler': obj_pose_relative_to_target_euler})
            target_pose_relative_to_target_euler = np.zeros(6)
            target_pose_relative_to_target_euler[3:] = target_pose_relative_to_target[:3]
            target_pose_relative_to_target_euler[3:] = euler_from_quaternion(*target_pose_relative_to_target[3:])
            misc.update({'target_pose_relative_to_target_euler': target_pose_relative_to_target_euler})

        return misc


class MyEnvironmentPeract(Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def launch(self):
        if self._pyrep is not None:
            raise RuntimeError('Already called launch!')
        self._pyrep = PyRep()
        self._pyrep.launch(join(DIR_PATH, TTT_FILE), headless=self._headless)

        arm_class, gripper_class, _ = SUPPORTED_ROBOTS[
            self._robot_setup]

        # We assume the panda is already loaded in the scene.
        if self._robot_setup != 'panda':
            # Remove the panda from the scene
            panda_arm = Panda()
            panda_pos = panda_arm.get_position()
            panda_arm.remove()
            arm_path = join(DIR_PATH, 'robot_ttms', self._robot_setup + '.ttm')
            self._pyrep.import_model(arm_path)
            arm, gripper = arm_class(), gripper_class()
            arm.set_position(panda_pos)
        else:
            arm, gripper = arm_class(), gripper_class()

        self._robot = Robot(arm, gripper)
        if self._randomize_every is None:
            self._scene = MyScene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup)
        else:
            raise NotImplementedError
            # self._scene = MyDomainRandomizationScene(
            #     self._pyrep, self._robot, self._obs_config, self._robot_setup,
            #     self._randomize_every, self._frequency,
            #     self._visual_randomization_config,
            #     self._dynamics_randomization_config)

        self._action_mode.arm_action_mode.set_control_mode(self._robot)


from rlbench.noise_model import NoiseModel
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask


def get_rgb_depth(sensor: VisionSensor, get_rgb: bool, get_depth: bool,
                get_pcd: bool, rgb_noise: NoiseModel,
                depth_noise: NoiseModel, depth_in_meters: bool):
    rgb = depth = pcd = None
    if sensor is not None and (get_rgb or get_depth):
        sensor.handle_explicitly()
        if get_rgb:
            rgb = sensor.capture_rgb()
            if rgb_noise is not None:
                rgb = rgb_noise.apply(rgb)
            rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
        if get_depth or get_pcd:
            depth = sensor.capture_depth(depth_in_meters)
            if depth_noise is not None:
                depth = depth_noise.apply(depth)
        if get_pcd:
            depth_m = depth
            if not depth_in_meters:
                near = sensor.get_near_clipping_plane()
                far = sensor.get_far_clipping_plane()
                depth_m = near + depth * (far - near)
            pcd = sensor.pointcloud_from_depth(depth_m)
            if not get_depth:
                depth = None
    return rgb, depth, pcd

def get_mask(sensor: VisionSensor, masks_as_one_channel=True):
    masks_as_one_channel = True
    mask_fn = rgb_handles_to_mask if masks_as_one_channel else lambda x: x

    mask = None
    if sensor is not None:
        sensor.handle_explicitly()
        mask = mask_fn(sensor.capture_rgb())
    return mask


def get_seg_mask(obj_list, mask):
    mask_id_list = np.unique(mask)

    select_id_list = []
    for obj in obj_list:
        id = obj.get_handle()
        name = obj.get_object_name(obj.get_handle())
        if name == "cup3":
            id = 97 # !! not sure why we need this
        if id not in mask_id_list:
            id = id + 1 # !! It solves the inconsisencies between object id and mask id (my guess is the object id is rounded down while the mask id could be rounded up/down)
        # if id not in mask_id_list:
        #     print(f"object {name} ({id}) is not found in {mask_id_list}")
        select_id_list.append(id)
    
    # print("select_id_list", select_id_list)

    seg_mask = np.zeros_like(mask)
    for seg_id, obj_id in enumerate(select_id_list):
        seg_mask[mask == obj_id] = seg_id + 1 # start from 1
    return seg_mask
