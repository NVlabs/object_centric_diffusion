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

import glob
from pathlib import Path
import shutil
import copy
import zarr
import pickle
from PIL import Image
import numpy as np
from utils.collect_utils import collect_narr_function, save_zarr
from utils.pose_utils import get_rel_pose, euler_from_quaternion


# create
data_dict = []
cur_data_dict = {
    "total_count": 0,
    # "img_arrays": [],
    # "point_cloud_arrays": [],
    # "depth_arrays": [],
    "state_arrays": [],
    "state_in_world_arrays": [],
    "state_next_arrays": [],
    "state_next_in_world_arrays": [],
    "goal_arrays": [],
    "action_arrays": [],
    "progress_arrays": [],
    "progress_binary_arrays": [],
    "task_stage_arrays": [],
    "variation_arrays": [],
    "episode_ends_arrays": [],
    "lang_list": [],  # debug only
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('task_name', type=str)
parser.add_argument('dataset_path', default="/tmp/record3d/[task_name]/r3d/", type=str)

args = parser.parse_args()

task_name = args.task_name
print(task_name)
dataset_path = args.dataset_path
print(dataset_path)
r3d_path_list = glob.glob(os.path.join(dataset_path, '*.r3d'))
r3d_path_list.sort()

for r3d_file in r3d_path_list:
    print(r3d_file)

    extract_path = os.path.join(dataset_path, Path(r3d_file).stem)
    grasp_obj_poses = np.load(os.path.join(extract_path, f"grasp_object_poses.npy"))
    target_object_poses = np.load(os.path.join(extract_path, f"target_object_poses.npy"))
    target_pose = target_object_poses[0]    # !! do we need to update target pose?


    episode_keypoints = list(range(len(grasp_obj_poses)))

    # create poses_dict
    poses_dict = {
        "task_stage": [],
        "grasp_obj_pose": [],
        "grasp_obj_pose_relative_to_target": [],
        "target_obj_pose": [],
        "keypoint": [],
    }
    stage_idx = 0
    variation = 0 # !! only one variation
    count = 0
    for idx, obj_pose in enumerate(grasp_obj_poses):
        obj_pose_relative_to_target = get_rel_pose(target_pose, obj_pose) # !! be careful about the order (target, grasp_object)

        poses_dict["grasp_obj_pose"].append(obj_pose)
        poses_dict["grasp_obj_pose_relative_to_target"].append(obj_pose_relative_to_target)
        poses_dict["target_obj_pose"].append(target_pose)
        poses_dict["task_stage"].append(stage_idx)
        if idx in episode_keypoints:
            poses_dict["keypoint"].append(count)
        count += 1

    # 
    arrays_sub_dict_list, total_count_sub_list = collect_narr_function([poses_dict], None)
    assert len(arrays_sub_dict_list) == len(total_count_sub_list) == 1
    
    for arrays_sub_dict, total_count_sub in zip(arrays_sub_dict_list, total_count_sub_list):
        cur_data_dict["total_count"] += total_count_sub
        cur_data_dict["episode_ends_arrays"].append(copy.deepcopy(cur_data_dict["total_count"])) # the index of the last step of the episode    
        # img_arrays.extend(copy.deepcopy(img_arrays_sub))
        # point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
        # depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
        cur_data_dict["state_arrays"].extend(copy.deepcopy(arrays_sub_dict["state"]))
        cur_data_dict["state_in_world_arrays"].extend(copy.deepcopy(arrays_sub_dict["state_in_world"]))
        cur_data_dict["state_next_arrays"].extend(copy.deepcopy(arrays_sub_dict["state_next"]))
        cur_data_dict["state_next_in_world_arrays"].extend(copy.deepcopy(arrays_sub_dict["state_next_in_world"]))
        cur_data_dict["goal_arrays"].extend(copy.deepcopy(arrays_sub_dict["goal"]))
        cur_data_dict["action_arrays"].extend(copy.deepcopy(arrays_sub_dict["action"]))
        cur_data_dict["progress_arrays"].extend(copy.deepcopy(arrays_sub_dict["progress"]))
        cur_data_dict["progress_binary_arrays"].extend(copy.deepcopy(arrays_sub_dict["progress_binary"]))
        cur_data_dict["task_stage_arrays"].extend(np.array(arrays_sub_dict["task_stage"]).reshape(-1, 1))
        cur_data_dict["variation_arrays"].extend(np.full((len(arrays_sub_dict["action"]), 1), fill_value=variation))


data_dict.append(cur_data_dict)
assert len(data_dict) == 1

# collect from all process
state_arrays = []
state_in_world_arrays = []
state_next_arrays = []
state_next_in_world_arrays = []
goal_arrays = []
action_arrays = []
progress_arrays = []
progress_binary_arrays = []
task_stage_arrays = []
variation_arrays = []
episode_ends_arrays = []
for i in range(1):
    state_arrays.extend(data_dict[i]["state_arrays"])
    state_in_world_arrays.extend(data_dict[i]["state_in_world_arrays"])
    state_next_arrays.extend(data_dict[i]["state_next_arrays"])
    state_next_in_world_arrays.extend(data_dict[i]["state_next_in_world_arrays"])
    goal_arrays.extend(data_dict[i]["goal_arrays"])
    action_arrays.extend(data_dict[i]["action_arrays"])
    progress_arrays.extend(data_dict[i]["progress_arrays"])
    progress_binary_arrays.extend(data_dict[i]["progress_binary_arrays"])
    task_stage_arrays.extend(data_dict[i]["task_stage_arrays"])
    variation_arrays.extend(data_dict[i]["variation_arrays"])
    episode_ends_arrays.extend(data_dict[i]["episode_ends_arrays"])


# save zarr
save_dir = f"/tmp/record3d/{task_name}/zarr/"
os.makedirs(save_dir, exist_ok=True)
save_zarr(
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
)