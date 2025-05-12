# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from math import e
import os, shutil
import zarr
import numpy as np
from termcolor import cprint
from scipy.spatial.transform import Rotation as R
from utils.pose_utils import get_rel_pose, euler_from_quaternion, quaternion_from_euler, relative_to_target_to_world, calculate_action, calculate_goal_pose


def check_pose_diff(pose1, pose2, trans_threshold, rot_threshold):
    action = calculate_action(pose1, pose2)
    dist = np.linalg.norm(action[:3])
    
    rot = R.from_quat(pose1[3:]) # (x, y, z, w)
    rotvec1 = rot.as_rotvec()
    rot = R.from_quat(pose2[3:]) # (x, y, z, w)
    rotvec2 = rot.as_rotvec()
    angle_diff = rotvec2 - rotvec1
    
    return dist >= trans_threshold or np.max(angle_diff) >= rot_threshold



# clean keyframe
# def clean_keyframe(key_frame_idx_list, obj_pose_relative_to_target_list, dist_threshold=0.01):
#     new_frame_idx_list = []
#     for idx in range(len(key_frame_idx_list)-1):
#         previous_frame_idx = key_frame_idx_list[idx-1]
#         cur_frame_idx = key_frame_idx_list[idx]
#         next_frame_idx = key_frame_idx_list[idx+1]

#         dist = (obj_pose_relative_to_target_list[next_frame_idx] - obj_pose_relative_to_target_list[cur_frame_idx])
#         dist = np.linalg.norm(dist[:3])
#         # print(dist)

#         if dist >= dist_threshold:
#             new_frame_idx_list.append(cur_frame_idx)
#     return new_frame_idx_list

def clean_keyframe(key_frame_idx_list, obj_pose_relative_to_target_list, keypoint_list, trans_threshold=0.05, rot_threshold=np.radians(20)):
    new_frame_idx_list = []
    prev_frame_idx = key_frame_idx_list[0]
    new_frame_idx_list.append(key_frame_idx_list[0])    # add initial frame index

    cur_keypoints_idx = 1 # start from keypoint_list[1]
    
    for idx in range(1, len(key_frame_idx_list)-1): # ignore first and last frame
        cur_frame_idx = key_frame_idx_list[idx]
        far = check_pose_diff(
            obj_pose_relative_to_target_list[prev_frame_idx], obj_pose_relative_to_target_list[cur_frame_idx],
            trans_threshold=trans_threshold, rot_threshold=rot_threshold,
        )
        # if (idx in keypoint_list) or far:
        #     # print(cur_frame_idx, cur_keypoints_idx, keypoint_list[cur_keypoints_idx])
        #     if (idx in keypoint_list): # add keyframe
        #         new_frame_idx_list.append(cur_frame_idx)    # add current frame index
        #         prev_frame_idx = cur_frame_idx  # update previous frame index for next iterarion
        #         cur_keypoints_idx += 1
        #     else: # check if too close to the keypoint
        #         if cur_keypoints_idx <= len(keypoint_list) - 1:
        #             next_keyframe_idx = keypoint_list[cur_keypoints_idx]
        #             far2 = check_pose_diff(
        #                 obj_pose_relative_to_target_list[cur_frame_idx], obj_pose_relative_to_target_list[next_keyframe_idx],
        #                 trans_threshold=trans_threshold*0.8, rot_threshold=rot_threshold*0.8,
        #             )
        #             if far2:   # only add if not too close to next keypoint
        #                 new_frame_idx_list.append(cur_frame_idx)    # add current frame index
        #                 prev_frame_idx = cur_frame_idx  # update previous frame index for next iterarion
        #         else: # no more keypoint
        #             new_frame_idx_list.append(cur_frame_idx)    # add current frame index
        #             prev_frame_idx = cur_frame_idx  # update previous frame index for next iterarion
        if far:
            new_frame_idx_list.append(cur_frame_idx)    # add current frame index
            prev_frame_idx = cur_frame_idx  # update previous frame index for next iterarion


    # check if add final frame
    far3 = check_pose_diff(
        obj_pose_relative_to_target_list[new_frame_idx_list[-1]], obj_pose_relative_to_target_list[key_frame_idx_list[-1]],
        trans_threshold=trans_threshold*0.8, rot_threshold=rot_threshold*0.8,
    )
    if far3:   # only add if not too close to next keypoint
        new_frame_idx_list.append(key_frame_idx_list[-1])    # add last frame index
    return new_frame_idx_list


def collect_narr_function(poses_dict_list, lang, intermediate_frame_length=1):
    # per demo        
    # img_arrays_sub = []
    # point_cloud_arrays_sub = []
    # depth_arrays_sub = []
    # state_arrays_sub = []
    # state_in_world_arrays_sub = []
    # state_next_arrays_sub = []
    # goal_arrays_sub = []
    # action_arrays_sub = []
    # total_count_sub = 0

    arrays_sub_dict_list = []
    total_count_sub_list = []


    for stage_idx, poses_dict in enumerate(poses_dict_list):
        arrays_sub_dict = {
            "state": [],                # current grasp obj pose
            "state_in_world": [],       # current grasp obj pose in world coordinate
            "state_next": [],           # next grasp obj pose
            "state_next_in_world": [],  # current grasp obj pose in world coordinate
            "goal": [],                 # target obj pose in world coordinate
            "action": [],               # action (from current grasp obj pose to next grasp obj pose)
            "progress": [],
            "progress_binary": [],
            "task_stage": [],
        }
        total_count_sub = 0

        # check if stage is the same
        task_stage_list = poses_dict["task_stage"]
        for task_stage in task_stage_list:
            assert stage_idx == task_stage

        # get key frame idx
        obj_pose_relative_to_target_list = poses_dict["grasp_obj_pose_relative_to_target"]
        seq_length = len(obj_pose_relative_to_target_list)
        key_frame_idx_list = range(0, seq_length, intermediate_frame_length)

        keypoint_list = poses_dict["keypoint"]

        assert len(key_frame_idx_list) == len(obj_pose_relative_to_target_list) == len(task_stage_list)
        key_frame_idx_list = clean_keyframe(key_frame_idx_list, obj_pose_relative_to_target_list, keypoint_list)

        # get each frame by key frame idx
        action = None
        for idx in range(len(key_frame_idx_list)-1):

            previous_frame_idx = key_frame_idx_list[idx-1]
            cur_frame_idx = key_frame_idx_list[idx]
            next_frame_idx = key_frame_idx_list[idx+1]

            # debug only
            # if action is not None:
            #     # prev_state = obj_pose_relative_to_target_list[previous_frame_idx] # previous pose
            #     # current_state = obj_pose_relative_to_target_list[cur_frame_idx] # current pose
            #     # current_state2 = prev_state + action
            #     # print(current_state, current_state2)

            #     prev_state = state_arrays_sub[idx-1] # previous pose
            #     current_state = obj_pose_relative_to_target_list[cur_frame_idx] # current pose
            #     current_state2 = prev_state + action_arrays_sub[idx-1]
            #     print(current_state, current_state2)

            cur_state_in_world = poses_dict["grasp_obj_pose"][cur_frame_idx]
            cur_state_next_in_world = poses_dict["grasp_obj_pose"][next_frame_idx]
            cur_state = poses_dict["grasp_obj_pose_relative_to_target"][cur_frame_idx] # current pose
            cur_state_next = poses_dict["grasp_obj_pose_relative_to_target"][next_frame_idx] # next pose
            cur_goal = poses_dict["target_obj_pose"][cur_frame_idx] # goal pose
            cur_task_stage = poses_dict["task_stage"][cur_frame_idx]

            # euler
            # action = obj_pose_relative_to_target_list[next_frame_idx] - obj_pose_relative_to_target_list[cur_frame_idx] # action to next pose

            # quaternion
            cur_action = calculate_action(cur_state, cur_state_next)
            # goal_pose = calculate_goal_pose(obj_pose_relative_to_target_list[cur_frame_idx], action)
            # print(goal_pose, obj_pose_relative_to_target_list[next_frame_idx])
            # exit()

            # save data
            total_count_sub += 1
            # img_arrays_sub.append(input_obs_visual)
            # state_in_world_arrays_sub.append(obs_robot_state_in_world)
            # state_arrays_sub.append(obs_robot_state)
            # state_next_arrays_sub.append(obs_robot_state_next)
            # goal_arrays_sub.append(obs_goal_state)
            # action_arrays_sub.append(action)
            arrays_sub_dict["state"].append(cur_state)
            arrays_sub_dict["state_in_world"].append(cur_state_in_world)
            arrays_sub_dict["state_next"].append(cur_state_next)
            arrays_sub_dict["state_next_in_world"].append(cur_state_next_in_world)
            arrays_sub_dict["goal"].append(cur_goal)
            arrays_sub_dict["action"].append(cur_action)
            arrays_sub_dict["task_stage"].append(cur_task_stage)
            # point_cloud_arrays_sub.append(time_step.observation_pointcloud)
            # depth_arrays_sub.append(time_step.observation_depth)
    

        # add final frame
        cur_frame_idx = -1
        cur_state_in_world = poses_dict["grasp_obj_pose"][-1]
        cur_state_next_in_world = cur_state_in_world
        cur_state = poses_dict["grasp_obj_pose_relative_to_target"][-1] # current pose
        cur_state_next = cur_state # next pose
        cur_goal = poses_dict["target_obj_pose"][-1] # goal pose
        cur_task_stage = poses_dict["task_stage"][-1]
        cur_action = calculate_action(cur_state, cur_state_next)

        # save final frame data
        total_count_sub += 1
        # img_arrays_sub.append(input_obs_visual)
        # state_in_world_arrays_sub.append(obs_robot_state_in_world)
        # state_arrays_sub.append(obs_robot_state)
        # state_next_arrays_sub.append(obs_robot_state_next)
        # goal_arrays_sub.append(obs_goal_state)
        # action_arrays_sub.append(action)
        arrays_sub_dict["state"].append(cur_state)
        arrays_sub_dict["state_in_world"].append(cur_state_in_world)
        arrays_sub_dict["state_next"].append(cur_state_next)
        arrays_sub_dict["state_next_in_world"].append(cur_state_next_in_world)
        arrays_sub_dict["goal"].append(cur_goal)
        arrays_sub_dict["action"].append(cur_action)
        arrays_sub_dict["task_stage"].append(cur_task_stage)
        # point_cloud_arrays_sub.append(time_step.observation_pointcloud)
        # depth_arrays_sub.append(time_step.observation_depth)

        # add prgoress
        progress = np.linspace(0., 1., num=len(key_frame_idx_list)).reshape(-1, 1)
        arrays_sub_dict["progress"].extend(progress)
        progress_binary = np.zeros_like(progress)
        progress_binary[-1] = 1.
        progress_binary[-2] = 0.9
        progress_binary[-3] = 0.8
        if len(progress_binary) >= 4:
            progress_binary[-4] = 0.7
        if len(progress_binary) >= 5:
            progress_binary[-5] = 0.6
        if len(progress_binary) >= 6:
            progress_binary[-6] = 0.5
        arrays_sub_dict["progress_binary"].extend(progress_binary)


        arrays_sub_dict_list.append(arrays_sub_dict)
        total_count_sub_list.append(total_count_sub)

    return arrays_sub_dict_list, total_count_sub_list


def save_zarr(
        save_dir,
        state_arrays,
        state_in_world_arrays,
        state_next_arrays,
        state_next_in_world_arrays,
        goal_arrays,
        action_arrays,
        progress_arrays,
        progress_binary_arrays,
        task_stage_arrays,
        variation_arrays,
        episode_ends_arrays,
    ):
    # img_arrays = np.stack(img_arrays, axis=0)
    # if img_arrays.shape[1] == 3: # make channel last
    #     img_arrays = np.transpose(img_arrays, (0,2,3,1))
    state_arrays = np.stack(state_arrays, axis=0)
    state_in_world_arrays = np.stack(state_in_world_arrays, axis=0)
    state_next_arrays = np.stack(state_next_arrays, axis=0)
    state_next_in_world_arrays = np.stack(state_next_in_world_arrays, axis=0)
    goal_arrays = np.stack(goal_arrays, axis=0)
    # point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    # depth_arrays = np.stack(depth_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    progress_arrays = np.stack(progress_arrays, axis=0)
    progress_binary_arrays = np.stack(progress_binary_arrays, axis=0)
    task_stage_arrays = np.stack(task_stage_arrays, axis=0)
    variation_arrays = np.stack(variation_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)
    print(state_arrays.shape)
    print(state_in_world_arrays.shape)
    print(state_next_arrays.shape)
    print(state_next_in_world_arrays.shape)
    print(goal_arrays.shape)
    print(progress_arrays.shape)
    print(progress_binary_arrays.shape)
    print(task_stage_arrays.shape)
    print(variation_arrays.shape)
    print(episode_ends_arrays.shape)

    # debug only
    episode_start_index = 0
    for idx, episode_end_index in enumerate(episode_ends_arrays):
        cur_state_array = state_arrays[episode_start_index: episode_end_index]
        cur_state_in_world_array = state_in_world_arrays[episode_start_index: episode_end_index]
        cur_state_next_array = state_next_arrays[episode_start_index: episode_end_index]
        cur_state_next_in_world_array = state_next_in_world_arrays[episode_start_index: episode_end_index]
        cur_goal_array = goal_arrays[episode_start_index: episode_end_index]
        cur_action_array = action_arrays[episode_start_index: episode_end_index]
        cur_progress_array = progress_arrays[episode_start_index: episode_end_index]
        cur_task_stage_array = task_stage_arrays[episode_start_index: episode_end_index]
        cur_progress_binary_array = progress_binary_arrays[episode_start_index: episode_end_index]
        cur_variation_array = variation_arrays[episode_start_index: episode_end_index]

        action_max = np.max(cur_action_array[:, :3], axis=0)
        action_min = np.min(cur_action_array[:, :3], axis=0)
        action_mean = np.mean(cur_action_array[:, :3], axis=0)
        action_range = action_max - action_min

        state_max = np.max(cur_state_array[:, :3], axis=0)
        state_min = np.min(cur_state_array[:, :3], axis=0)
        state_mean = np.mean(cur_state_array[:, :3], axis=0)
        state_range = state_max - state_min

        progress_max = np.max(cur_progress_array[:, :3], axis=0)
        progress_min = np.min(cur_progress_array[:, :3], axis=0)
        progress_mean = np.mean(cur_progress_array[:, :3], axis=0)
        progress_range = progress_max - progress_min
        np.set_printoptions(precision=3)
        print(f"Episode {idx}: length {len(cur_action_array)}, action range {action_range}, state range {state_range}, progress range {progress_range}")
        # update start index for next episode
        episode_start_index = episode_end_index

    # save zarr by collecting across all demos
    # create zarr file
    cprint(f'Saved zarr file to {save_dir}', 'green')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)  # remove dir and all contains
    os.makedirs(save_dir, exist_ok=True)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    # img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    state_chunk_size = (100, state_arrays.shape[1])
    state_in_world_chunk_size = (100, state_in_world_arrays.shape[1])
    state_next_chunk_size = (100, state_next_arrays.shape[1])
    state_next_in_world_chunk_size = (100, state_next_in_world_arrays.shape[1])
    goal_chunk_size = (100, goal_arrays.shape[1])
    # full_state_chunk_size = (100, full_state_arrays.shape[1])
    # point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    # depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    progress_chunk_size = (100, progress_arrays.shape[1])
    progress_binary_chunk_size = (100, progress_binary_arrays.shape[1])
    task_stage_chunk_size = (100, task_stage_arrays.shape[1])
    variation_chunk_size = (100, variation_arrays.shape[1])
    # zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state_in_world', data=state_in_world_arrays, chunks=state_in_world_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state_next', data=state_next_arrays, chunks=state_next_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state_next_in_world', data=state_next_in_world_arrays, chunks=state_next_in_world_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('goal', data=goal_arrays, chunks=goal_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('full_state', data=full_state_arrays, chunks=full_state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('progress', data=progress_arrays, chunks=progress_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('progress_binary', data=progress_binary_arrays, chunks=progress_binary_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('task_stage', data=task_stage_arrays, chunks=task_stage_chunk_size, dtype='int64', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('variation', data=variation_arrays, chunks=variation_chunk_size, dtype='int64', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

    # cprint(f'-'*50, 'cyan')
    # # print shape
    # cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    # cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    # cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    # cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    # cprint(f'full_state shape: {full_state_arrays.shape}, range: [{np.min(full_state_arrays)}, {np.max(full_state_arrays)}]', 'green')
    # cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    # cprint(f'Saved zarr file to {save_dir}', 'green')
