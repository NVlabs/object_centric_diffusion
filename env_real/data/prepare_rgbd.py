# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
import cv2
import os
import argparse
import liblzfse
import open3d as o3d
import json
import imageio


def load_depth(filepath, H, W):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)
        depth_img = depth_img.reshape((H, W)) 
    return depth_img

def load_conf(filepath, H, W):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        conf = np.frombuffer(decompressed_bytes, dtype=np.int8)
        conf = conf.reshape((H, W))
    return np.float32(conf)

def create_point_cloud_depth(depth, rgb, fx, fy, cx, cy):
    depth_shape = depth.shape
    [x_d, y_d] = np.meshgrid(range(0, depth_shape[1]), range(0, depth_shape[0]))
    x3 = np.divide(np.multiply((x_d-cx), depth), fx)
    y3 = np.divide(np.multiply((y_d-cy), depth), fy)
    z3 = depth

    coord =  np.stack((x3, y3, z3), axis=2)

    rgb_norm = rgb/255

    return np.concatenate((coord, rgb_norm), axis=2)

if __name__ == '__main__':
    from pathlib import Path
    from zipfile import ZipFile
    import glob
    import shutil

    from scipy.spatial.transform import Rotation

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

        # unzip file
        extract_path = os.path.join(dataset_path, Path(r3d_file).stem)
        ZipFile(r3d_file).extractall(extract_path)

        # get metadata
        with open(os.path.join(extract_path, 'metadata'), "rb") as f:
            metadata = json.loads(f.read())
        poses = np.asarray(metadata['poses'])   # (N, 7) [x, y, z, qx, qy, qz, qw]

        # get intrinsics
        K = np.asarray(metadata['K']).reshape(3, 3).T
        np.savetxt(os.path.join(extract_path, 'K.txt'), K)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # process data
        rgb_path_list = glob.glob(os.path.join(extract_path, 'rgbd', '*.jpg'))
        assert len(poses) == len(rgb_path_list), f"{len(poses)} {len(rgb_path_list)}"

        # get HW
        downscale_factor = 1. # 1. for front and 3.75 for rear camera set by ios
        H,W = cv2.imread(os.path.join(extract_path, 'rgbd', f'0.jpg')).shape[:2]
        H_dc, W_dc = int(H/downscale_factor), int(W/downscale_factor)

        writer = imageio.get_writer(os.path.join(extract_path, 'video.mp4'), fps=30)
        for frame_idx in range(len(rgb_path_list)):
            
            rgb_path = os.path.join(extract_path, 'rgbd', f'{frame_idx}.jpg')
            fname = Path(rgb_path).stem
            writer.append_data(imageio.imread(rgb_path))

            # copy all rgb to dir "/rgb"
            rgb_dir = os.path.join(extract_path, 'rgb')
            os.makedirs(rgb_dir, exist_ok=True)
            shutil.copy(rgb_path, os.path.join(rgb_dir, fname + '.jpg'))

            # save depth
            depth_path = rgb_path.replace('.jpg', '.depth')
            depth = load_depth(str(depth_path), H_dc, W_dc)

            depth_dir = os.path.join(extract_path, 'depth')
            os.makedirs(depth_dir, exist_ok=True)
            cv2.imwrite(os.path.join(depth_dir, fname + '.png'), (depth * 1000.).astype(np.uint16))
            # print(np.max(np.nan_to_num(depth)), np.min(np.nan_to_num(depth)))
            # depth = cv2.imread(os.path.join(rgb_dir, fname + '.png'), cv2.IMREAD_ANYDEPTH) / 1000.
            # print(np.max(depth), np.min(depth))

            # save extrinsic
            pose = poses[frame_idx]   # cam2world
            pose_mat = np.eye(4)
            pose_mat[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
            pose_mat[3, :3] = pose[:3]
            extrinsic = np.linalg.inv(pose_mat)

            pose_dir = os.path.join(extract_path, 'cam_in_ob')
            os.makedirs(pose_dir, exist_ok=True)
            np.savetxt(os.path.join(pose_dir, fname + '.txt'), extrinsic)

            if frame_idx == 0:
                # load rgb
                rgb_path = os.path.join(extract_path, 'rgbd', fname+'.jpg')
                rgb = cv2.imread(rgb_path)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                # load confidence
                conf_path = os.path.join(extract_path, 'rgbd', fname+'.conf')
                if os.path.exists(conf_path):
                    conf = load_conf(conf_path, H_dc, W_dc)
                else:
                    conf = None

                # get pointcloud (interpolated)
                depth_resized = cv2.resize(depth, (W, H))
                pc = create_point_cloud_depth(depth_resized, rgb, fx, fy, cx, cy).reshape(-1, 6)
                if conf is not None:
                    conf_resized = cv2.resize(conf, (W, H), cv2.INTER_NEAREST_EXACT)
                    pc = pc[conf_resized.reshape(-1) >= 2]
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:])
                # o3d.visualization.draw_geometries([pcd])
                o3d.io.write_point_cloud(f'{extract_path}/pc_interpolated.ply', pcd)

                pcd.transform(extrinsic)
                o3d.io.write_point_cloud(f'{extract_path}/pc_interpolated_in_world.ply', pcd)

                # get pointcloud (original resolution)
                rgb_resized = cv2.resize(rgb, (W_dc, H_dc))
                pc = create_point_cloud_depth(depth, rgb_resized, fx / downscale_factor, fy / downscale_factor, cx / downscale_factor, cy / downscale_factor).reshape(-1, 6)
                if conf is not None:
                    pc = pc[conf.reshape(-1) >= 2]
                
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(pc[:, :3])
                pcd2.colors = o3d.utility.Vector3dVector(pc[:, 3:])
                # o3d.visualization.draw_geometries([pcd, pcd2])
                o3d.io.write_point_cloud(f'{extract_path}/pc_original.ply', pcd2)

                pcd2.transform(extrinsic)
                o3d.io.write_point_cloud(f'{extract_path}/pc_original_in_world.ply', pcd2)
        writer.close()