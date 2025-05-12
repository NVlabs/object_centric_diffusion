# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import sys

from networkx import degree
sys.path.insert(0, os.getcwd())

import copy
import torch
import numpy as np
import einops

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply

from pyrep.objects import Dummy, VisionSensor, Shape
from env_rlbench.policy.subgoal_policy import RLBenchSubGoalPolicy
from utils.pose_utils import calculate_action, calculate_goal_pose, get_rel_pose, euler_from_quaternion, quaternion_from_euler, relative_to_target_to_world


class RLBenchDP3Policy(RLBenchSubGoalPolicy):
    def __init__(self, env, sub_goal_policy: BasePolicy, use_fp, enable_stage):
        self.env = env

        # initial stage
        self.stage = "reach" 

        # timer for stages
        self.T_GRASP = 10 
        self.GRASP_TIMER = 0 # closing gripper -> moving
        self.MOVEUP_TIMER = 0

        # dp3
        self.sub_goal_policy = sub_goal_policy
        self.device = sub_goal_policy.device
        self.dtype = sub_goal_policy.dtype
        self.cur_subgoal = None
        self.cur_subgoal_obj = None
        self.cur_progress = None
        self.history = [] # for dp3
        self.n_obs_steps = sub_goal_policy.n_obs_steps
        self.use_fp = use_fp

        self.enable_stage = enable_stage
        self.cur_loop_num = 0
        self.max_loop_num = 1

        self.predict_type = 'rel'
        self.history_dist_threshold = 0.001 #0.05 # !! Must be greater than moving distance otherwise subgoal would not be updated
        assert self.n_obs_steps == 1
    
    def reset_in_loop(self):
        # initial stage
        self.stage = "reach" 

        # timer for stages
        self.T_GRASP = 10 
        self.GRASP_TIMER = 0 # closing gripper -> moving
        self.MOVEUP_TIMER = 0

        # dp3
        self.cur_subgoal = None
        self.cur_subgoal_obj = None
        self.cur_progress = None
        self.history = [] # for dp3

    def reset(self):
        self.reset_in_loop()

        # visualization
        self.ee_pose_history = []
        self.obj_pose_history = []
        self.target_obj_pose_history = []
        self.subgoal_obj_pose_history = []  # pred
        self.subgoal_ee_pose_history = []  # pred
        self.buffer_pose_history = []   # history buffer

        # reset loop for multi-stage task
        self.cur_loop_num = 0
        self.max_loop_num = 1

    def _get_current_pose(self, obs):
        if self.enable_stage:
            target_obj_pose = obs.misc[f"stage{self.cur_loop_num}_target_obj_pose"]
        else:
            target_obj_pose =  obs.misc["target_obj_pose"]

        # TODO: add last mile support
        # if self.env._last_mile:
        #     grasp_obj_pose = self.env.current_pose_estimation
        # else:
        #     grasp_obj_pose = obs.misc["grasp_obj_pose"]

        if self.use_fp:
            grasp_obj_pose = self.env.current_pose_estimation
        else:
            if self.enable_stage:
                grasp_obj_pose = obs.misc[f"stage{self.cur_loop_num}_grasp_obj_pose"]
            else:
                grasp_obj_pose = obs.misc["grasp_obj_pose"]
        
        DEBUG = False
        if self.use_fp and DEBUG:
            from scipy.spatial.transform import Rotation as R
            def get_euler(pose):
                rot = R.from_quat(pose[3:]) # (x, y, z, w)
                euler = rot.as_euler('xyz', degrees=True)
                return euler
            def get_rotvec(pose):
                rot = R.from_quat(pose[3:]) # (x, y, z, w)
                rotvec = rot.as_rotvec('xyz')
                return rotvec
            
            grasp_obj_pose = self.env.current_pose_estimation
            euler_fp = get_euler(grasp_obj_pose)
            grasp_obj_pose_gt = obs.misc["grasp_obj_pose"]
            euler_gt = get_euler(grasp_obj_pose_gt)
            print("fp", grasp_obj_pose, euler_fp)
            print("gt", grasp_obj_pose_gt, euler_gt)

            debug_dir = "./debug"

            # save mesh in simulation's global frame (converted from FP) - #*the mesh should align with fp_pcd
            world_mat = np.eye(4)
            world_mat[:3, 3] = grasp_obj_pose[:3]
            world_mat[:3, :3] = R.from_quat(grasp_obj_pose[3:]).as_matrix()
            m = self.env.pose_estimator.mesh.copy()
            m.apply_transform(world_mat)
            m.export(f'{debug_dir}/model_tf_world_fp.obj')

            # save mesh in simulation's global frame (ground truth) - #*the mesh should align with fp_pcd
            world_mat = np.eye(4)
            world_mat[:3, 3] = grasp_obj_pose_gt[:3]
            world_mat[:3, :3] = R.from_quat(grasp_obj_pose_gt[3:]).as_matrix()
            m = self.env.pose_estimator.mesh.copy()
            m.apply_transform(world_mat)
            m.export(f'{debug_dir}/model_tf_world_gt.obj')

            exit()
        return grasp_obj_pose, target_obj_pose

    def _get_sub_goal(self, obs, lang_token_embs=None, stage_embs=None):
        # get the position and quaternion of object
        scene = self.env.env._scene

        obs_dict = {}
        obs_dict['agent_pos'] = np.array(self.history)

        # add language
        if lang_token_embs is not None:
            obs_dict['lang_token_embs'] = lang_token_embs
        if stage_embs is not None:
            obs_dict['stage_embs'] = stage_embs

        # get action
        np_obs_dict = dict(obs_dict)
        obs_dict = dict_apply(np_obs_dict,
                                lambda x: torch.from_numpy(x).to(
                                    device=self.device))


        with torch.no_grad():
            obs_dict_input = {}
            # obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).float()
            if lang_token_embs is not None:
                obs_dict_input['lang_token_embs'] = obs_dict['lang_token_embs'].unsqueeze(0).float()
            if stage_embs is not None:
                obs_dict_input['stage_embs'] = obs_dict['stage_embs'].unsqueeze(0).float()
            action_dict = self.sub_goal_policy.predict_action(obs_dict_input)

        np_action_dict = dict_apply(action_dict,
                                    lambda x: x.detach().to('cpu').numpy())
        pred = np_action_dict['action'].squeeze(0)
        select_pred = pred[0]
        select_action = select_pred[:7]
        select_progress = select_pred[7]

        # get current pose
        gripper_pose = self.env._task._robot.arm.get_tip().get_pose()        
        _, target_obj_pose = self._get_current_pose(obs)
        grasp_obj_pose = relative_to_target_to_world(np.array(self.history[0]), target_obj_pose)
        obj_pose_relative_to_target = get_rel_pose(target_obj_pose, grasp_obj_pose)

        if self.predict_type == 'rel':
            # quaternion
            subgoal_relative_to_target = calculate_goal_pose(obj_pose_relative_to_target, select_action)
        elif self.predict_type == 'abs':
            # quaternion (abolute)
            subgoal_relative_to_target = select_action
        else:
            raise NotImplementedError
        
        subgoal_gripper = self._subgoal_relative_to_target_to_subgoal_gripper(subgoal_relative_to_target, target_obj_pose)
        return subgoal_gripper, subgoal_relative_to_target, select_progress

    def _update_history_buffer(self, obs):
        # get the position and quaternion of object
        scene = self.env.env._scene

        # get current pose
        gripper_pose = self.env._task._robot.arm.get_tip().get_pose()
        grasp_obj_pose, target_obj_pose = self._get_current_pose(obs)

        obj_pose_relative_to_target = get_rel_pose(target_obj_pose, grasp_obj_pose)

        # # convert from quaternion to euler
        # obj_pose_relative_to_target_euler = np.zeros(6)
        # obj_pose_relative_to_target_euler[:3] = obj_pose_relative_to_target[:3]
        # obj_pose_relative_to_target_euler[3:] = euler_from_quaternion(*obj_pose_relative_to_target[3:])

        assert self.n_obs_steps > 0
        if len(self.history) < self.n_obs_steps:
            self.history.extend([obj_pose_relative_to_target for _ in range(self.n_obs_steps-len(self.history))])
        else:
            dist = (obj_pose_relative_to_target - self.history[-1])
            dist = np.linalg.norm(dist[:3])
            if dist >= self.history_dist_threshold:
                # TODO: update history buffer with arbitrary length
                if self.n_obs_steps == 1:
                    self.history[0] = obj_pose_relative_to_target
                elif self.n_obs_steps == 2:
                    self.history[0] = self.history[1]
                    self.history[1] = obj_pose_relative_to_target
                elif self.n_obs_steps == 4:
                    self.history[0] = self.history[1]
                    self.history[1] = self.history[2]
                    self.history[2] = self.history[3]
                    self.history[3] = obj_pose_relative_to_target
                else:
                    raise NotImplementedError

    def get_action(self, obs, lang_token_embs=None, stage_embs=None):
        assert self.stage in ["reach", "grasp", "move", "leave", "end"]
        if self.env._start_tracking:
            grasp_obj_pose, target_obj_pose = self._get_current_pose(obs)
            gripper_pose = self.env._task._robot.arm.get_tip().get_pose()
            self.T_obj_to_gripper = get_rel_pose(grasp_obj_pose, gripper_pose)
        task_name = self.env.env._scene.task.get_name()
        gripper_pose = self.env._task._robot.arm.get_tip().get_pose()
        
        # get action
        if self.stage == "reach":
            if task_name == "stack_cups":
                self.env.run_demo_until_n_waypoint(start=5*self.cur_loop_num, n_waypoint=5*self.cur_loop_num+2)
            else:
                self.env.run_demo_until_n_waypoint(n_waypoint=2)

            # start tracking only after grasping stage
            self.env.start_tracking()
            self.env._rlbench_env._scene.step()

            self.stage = "grasp"
            return None
        
        elif self.stage == "grasp":
            pass    # included in "reach" stage

            grasp_obj_pose, target_obj_pose = self._get_current_pose(obs)
            gripper_pose = self.env._task._robot.arm.get_tip().get_pose()
            self.T_obj_to_gripper = get_rel_pose(grasp_obj_pose, gripper_pose)
            # self.T_gripper_to_obj = get_rel_pose(gripper_pose, grasp_obj_pose)
            # self.env.end_tracking()

            self.stage = "move"
            return None
        
        elif self.stage == "move":
            if self.cur_progress is not None and self.cur_progress > 0.9 and (task_name not in ["reach_and_drag"]):
                # move to next stage
                self.stage = "leave"
                return None
                
            subgoal_threshold = 0.05 # distance between current EE pose and subgoal pose
            if self.cur_subgoal is not None:
                dist = np.linalg.norm(self.cur_subgoal[:3] - gripper_pose[:3])
                angle_diff = np.linalg.norm(self.cur_subgoal[3:] - gripper_pose[3:])
                angle_diff2 = np.linalg.norm(-self.cur_subgoal[3:] - gripper_pose[3:])
                gripper_goal_far = dist > subgoal_threshold or min(angle_diff, angle_diff2) > 0.001

            # get first subgoal
            if self.cur_subgoal is None:
                self._update_history_buffer(obs)
                # self._update_history_buffer_gt()
                self.cur_subgoal, self.cur_subgoal_obj, self.cur_progress = self._get_sub_goal(obs, lang_token_embs, stage_embs)
            # get new/next subgoal
            elif not gripper_goal_far:
                self._update_history_buffer(obs)
                # self._update_history_buffer_gt()
                self.cur_subgoal, self.cur_subgoal_obj, self.cur_progress = self._get_sub_goal(obs, lang_token_embs, stage_embs)
            # use previous subgoal
            else:
                pass
            goal_pose = self.cur_subgoal
            goal_obj_pose = self.cur_subgoal_obj
            return self._move_to(goal_pose, close_gripper=True, object_centric=True)

        elif self.stage == "leave":
            self.env.end_tracking()
            if self.GRASP_TIMER < self.T_GRASP:
                self.GRASP_TIMER += 1
                return self._open_gripper()
            else:
                self.GRASP_TIMER = 0 # reset timer
                self.stage = "end"
                return self._move_away(axis='z', dist=0.1)

        elif self.stage == "end":

            if self.cur_loop_num < self.max_loop_num - 1:
                # grasp next object
                if task_name == "place_cups":
                    # self.env.env._scene.task._cups_placed += 1
                    pass
                elif task_name == "stack_cups":
                    pass # handled elsewhere
                elif task_name == "stack_blocks":
                    self.env.env._scene.task.blocks_stacked += 1
                else:
                    raise NotImplementedError
                # start a new loop
                self.cur_loop_num += 1
                self.reset_in_loop()
                self.env.move_to_next_stage()
            else:
                # ends
                print("Return empty action. (current policy stage: %s)" % self.stage)
                return None
        else:
            raise NotImplementedError