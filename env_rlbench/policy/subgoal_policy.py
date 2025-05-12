# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from utils.pose_utils import get_rel_pose, relative_to_target_to_world


class RLBenchSubGoalPolicy:
    def __init__(self, env):
        pass
    
    def _open_gripper(self):
        gripper = self.env._rlbench_env._scene.robot.gripper        
        scene = self.env._rlbench_env._scene
        gripper.release() # remove object from the list gripper._grasped_objects
    
        done = False
        i = 0
        vel = 0.04
        open_amount = 1.0 #if gripper_name == 'Robotiq85Gripper' else 0.8
        while not done:
            done = gripper.actuate(open_amount, velocity=vel)
            scene.step()
            i += 1
            if i > 1000:
                self.fail('Took too many steps to open')
        return None
    
    def _move_away(self, axis='z', dist=0.2):
        gripper_pose = self.env._task._robot.arm.get_tip().get_pose()
        action = np.zeros(8)
        action[7] = 1 # open gripper
        action[:7] = gripper_pose
        if 'x' in axis:
            action[0] += dist[0] if isinstance(dist, list) else dist
        if 'y' in axis:
            action[1] += dist[1] if isinstance(dist, list) else dist
        if 'z' in axis:
            action[2] += dist[2] if isinstance(dist, list) else dist
        return action
    
    def _move_to(self, goal_pose, close_gripper=True, object_centric=False):
        action = np.zeros(8)
        action[7] = 0 # close gripper
        action[:7] = goal_pose # assume the action mode is EndEffectorPoseViaIK or EndEffectorPoseViaPlanning
        return action

    def _subgoal_relative_to_target_to_subgoal_gripper(self, subgoal_relative_to_target, target_obj_pose):
        subgoal = relative_to_target_to_world(subgoal_relative_to_target, target_obj_pose)
        subgoal_gripper =  relative_to_target_to_world(self.T_obj_to_gripper, subgoal)
        return subgoal_gripper