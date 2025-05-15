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

from foundation_pose.wrapper import FoundationPoseWrapper
from pathlib import Path
import glob
import numpy as np
import os
import imageio
import cv2
from PIL import Image
import trimesh
from foundation_pose.Utils import depth2xyzmap, draw_posed_3d_box, draw_xyz_axis, toOpen3dCloud, trimesh_add_pure_colored_texture
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import open3d as o3d
from env_real.utils.realworld_objects import real_task_object_dict


def get_vis_pose(pose, color, K, mesh):
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    center_pose = pose @ np.linalg.inv(to_origin)

    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
    return vis

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task_name', type=str)
    args = parser.parse_args()
    task_name = args.task_name
    print(task_name)
    dataset_path = f"/tmp/record3d/{task_name}/r3d/"
    r3d_path_list = glob.glob(os.path.join(dataset_path, '*.r3d'))
    r3d_path_list.sort()
    

    for obj_type in ['grasp', 'target']:
        
        # get mesh
        obj_name = real_task_object_dict[task_name][f"{obj_type}_object_name"]
        mesh_path = f"/tmp/record3d/mesh/{obj_name}/{obj_name}.obj"
        mesh = trimesh.load(mesh_path) 
        mesh.vertices = mesh.vertices - np.mean(mesh.vertices, axis=0)
        # mesh.show()

        for r3d_file in r3d_path_list:
            print(r3d_file)

            # if not "2024-09-02--19-49-49" in r3d_file:
            #     continue

            # get FP
            pose_estimation_wrapper = FoundationPoseWrapper(mesh_dir=None)
            pose_estimation_wrapper.mesh = mesh
            pose_estimator = pose_estimation_wrapper.create_estimator(debug_level=0)

            # process data
            extract_path = os.path.join(dataset_path, Path(r3d_file).stem)
            rgb_path_list = glob.glob(os.path.join(extract_path, 'rgb', '*.jpg'))
            vis_pose_list = []
            object_poses = []

            # get K
            K_path = os.path.join(extract_path, 'K.txt')
            K = np.loadtxt(K_path).reshape(3,3)

            # get HW
            H,W = cv2.imread(os.path.join(extract_path, 'rgb', f'0.jpg')).shape[:2]

            for frame_idx in tqdm(range(len(rgb_path_list))):

                # get rgb
                rgb_path = os.path.join(extract_path, 'rgb', f'{frame_idx}.jpg')
                fname = Path(rgb_path).stem
                rgb = imageio.imread(rgb_path)[...,:3]
                # rgb = cv2.resize(rgb, (192,256), interpolation=cv2.INTER_NEAREST)

                # get depth
                depth_path = os.path.join(extract_path, 'depth', f'{frame_idx}.png')
                depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / 1000.
                depth = cv2.resize(depth, (W, H))
                
                if frame_idx == 0:
                    # get mask
                    mask_path = os.path.join(extract_path, f'mask_{obj_type}', f'{frame_idx}.png')
                    mask_image = Image.open(mask_path).convert('P')
                    mask = np.array(mask_image)
                    # mask = cv2.resize(mask, (192,256), interpolation=cv2.INTER_NEAREST)
                    mask = (mask == 1).astype(bool).astype(np.uint8)

                    fp_mat = pose_estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
                else:
                    fp_mat = pose_estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=2)
                
                # get pose in world frame
                extrinsic_path = os.path.join(extract_path, 'cam_in_ob', fname + '.txt')
                extrinsic = np.loadtxt(extrinsic_path).reshape(4, 4)
                pose_in_world =  np.matmul(extrinsic, fp_mat)
                pose_in_world_quat = np.concatenate([
                    pose_in_world[:3, 3],
                    Rotation.from_matrix(pose_in_world[:3, :3]).as_quat(),
                ])
                object_poses.append(pose_in_world_quat)

                # visualization
                vis_pose = get_vis_pose(
                    pose=fp_mat, 
                    color=rgb, 
                    K=K, 
                    mesh=pose_estimation_wrapper.mesh
                )
                vis_pose_list.append(vis_pose)
            
            # save object pose in world
            assert len(object_poses) == len(rgb_path_list), f"{len(object_poses)} {len(rgb_path_list)}"
            np.save(os.path.join(extract_path, f"{obj_type}_object_poses.npy"), np.array(object_poses))
            object_poses = np.load(os.path.join(extract_path, f"{obj_type}_object_poses.npy"))

            # save video
            video_path = os.path.join(extract_path, 'vis', f'pose_track_{obj_type}.mp4')
            video_writer = imageio.get_writer(video_path, fps=40)
            for img in vis_pose_list:
                video_writer.append_data(img)
            video_writer.close()

    # save point cloud in target's frame
    for r3d_file in r3d_path_list:
        print(r3d_file)
        # save scene_in_target pointcloud
        pcd = o3d.io.read_point_cloud(f'{extract_path}/pc_original_in_world.ply')
        points = np.asarray(pcd.points)
        print(points.shape)

        from utils.pose_utils import get_rel_pose, euler_from_quaternion, compute_rel_transform
        import utils.transform_utils as T
        grasp_obj_pose = np.load(os.path.join(extract_path, f"grasp_object_poses.npy"))[0]
        target_obj_pose = np.load(os.path.join(extract_path, f"target_object_poses.npy"))[0]
        points = np.vstack([
                compute_rel_transform(target_obj_pose[:3], T.quat2mat(target_obj_pose[3:]), p[:3], T.quat2mat(grasp_obj_pose[3:]))[0]
                for p in points
            ]
        )
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        o3d.io.write_point_cloud(f"{extract_path}/scene_in_target.ply", pcd)