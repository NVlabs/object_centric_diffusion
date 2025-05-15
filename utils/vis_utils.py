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

import torch
import numpy as np
from utils.vis_o3d_utils import O3DVisualizer
from utils.pose_utils import calculate_goal_pose


def dp3_visualize(agent_pos, pred=None, target=None, visualize=True, predict_type='relative'):
    if visualize:
        vis = O3DVisualizer()

    if isinstance(agent_pos, torch.Tensor):
        agent_pos_cpu = agent_pos.clone().detach().cpu().numpy()
        if pred is not None:
            pred_cpu = pred.clone().detach().cpu().numpy()
        if target is not None:
            target_cpu = target.clone().detach().cpu().numpy()
    else:
        agent_pos_cpu = agent_pos
        if pred is not None:
            pred_cpu = pred
        if target is not None:
            target_cpu = target
    
    if len(agent_pos_cpu.shape) == 2: # not a batch
        agent_pos_cpu = agent_pos_cpu[None, ...]
        if pred is not None:
            pred_cpu = pred_cpu[None, ...]
        if target is not None:
            target_cpu = target_cpu[None, ...]
    print(agent_pos_cpu.shape)


    input_count = 0
    pred_count = 0
    gt_count = 0
    for batch_idx in range(len(agent_pos_cpu)):

        cur_pose = agent_pos_cpu[batch_idx]
        print(cur_pose.shape)
        if target is not None:
            gt_pose = target_cpu[batch_idx]
            print(gt_pose.shape)


        for pose_idx in range(cur_pose.shape[0]):
            np.set_printoptions(precision=3)
            print("batch %d" % batch_idx)
            # print("input", cur_pose[pose_idx])
            if visualize:
                vis.add_pose_from_traj(cur_pose[pose_idx].reshape(-1, 7), pos_only=False, paint_color=[1., 0., 0.])
            input_count += 1

            if target is not None:
                if predict_type == 'relative':
                    # euler
                    # cur_gt_pose = cur_pose[pose_idx] + gt_pose[pose_idx]
                    # quaternion
                    cur_gt_pose = calculate_goal_pose(cur_pose[pose_idx][:7], gt_pose[pose_idx][:7])
                else:
                    cur_gt_pose = gt_pose[pose_idx]
                # print("action (gt)", cur_gt_pose)
                if visualize:
                    vis.add_pose_from_traj(cur_gt_pose.reshape(-1, 7), pos_only=False, paint_color=[0., 1., 0.])
                gt_count += 1
        if pred is not None:
            pred_pose = pred_cpu[batch_idx]
            print(pred_pose.shape)

            if predict_type == 'relative':
                iterated_pose = cur_pose[0]
            for pose_idx in range(pred_pose.shape[0]):
                if predict_type == 'relative':
                    # euler
                    # cur_pred_pose = cur_pose[pose_idx] + pred_pose[pose_idx]
                    # quaternion
                    # cur_pred_pose = calculate_goal_pose(cur_pose[pose_idx], pred_pose[pose_idx])
                    cur_pred_pose = calculate_goal_pose(iterated_pose, pred_pose[pose_idx])
                else:
                    cur_pred_pose = pred_pose[pose_idx]
                # print("action (pred)", cur_pred_pose)
                if visualize:
                    vis.add_pose_from_traj(cur_pred_pose.reshape(-1, 7), pos_only=False, paint_color=[0., 0., 1.])

                if predict_type == 'relative':
                    # print("iterated_pose", iterated_pose)
                    # print("cur_pose", cur_pose[pose_idx+1])
                    if pose_idx+1 < cur_pose.shape[0]:
                        print("difference", iterated_pose - cur_pose[pose_idx+1])
                    iterated_pose = cur_pred_pose
                pred_count += 1
    
        print(input_count, gt_count, pred_count)
        if visualize:
            vis.draw()


import trimesh
from foundation_pose.Utils import draw_posed_3d_box, draw_xyz_axis

def get_vis_pose(pose, color, K, mesh):
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    center_pose = pose @ np.linalg.inv(to_origin)

    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
    return vis