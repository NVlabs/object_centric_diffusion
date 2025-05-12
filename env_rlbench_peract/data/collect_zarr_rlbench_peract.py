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

from multiprocessing import Process, Manager

from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task

import os
import copy
import pickle
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np

from absl import app
from absl import flags

from env_rlbench_peract.utils.rlbench_utils import MyEnvironmentPeract
from utils.collect_utils import collect_narr_function, save_zarr
from utils.collect_utils_rlbench import keypoint_discovery


FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    '/tmp/rlbench_data/',
                    'Where to save the demos.')
flags.DEFINE_string('peract_demo_dir', 
                    '/tmp/peract/raw/', 
                    'Where to load the peract demos.')
flags.DEFINE_string('split',
                    '',
                    'train/val/test set')
flags.DEFINE_list('tasks', [],
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [512, 512],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl3', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 10,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_bool('all_variations', True,
                  'Include all variations when sampling epsiodes')


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_demo(demo, example_path, variation):

    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(
        example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(
        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(
        example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(
        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(
        example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(
        example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(
        example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(
        example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    check_and_make(example_path)
    check_and_make(left_shoulder_rgb_path)
    check_and_make(left_shoulder_depth_path)
    check_and_make(left_shoulder_mask_path)
    check_and_make(right_shoulder_rgb_path)
    check_and_make(right_shoulder_depth_path)
    check_and_make(right_shoulder_mask_path)
    check_and_make(overhead_rgb_path)
    check_and_make(overhead_depth_path)
    check_and_make(overhead_mask_path)
    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)
    check_and_make(wrist_mask_path)
    check_and_make(front_rgb_path)
    check_and_make(front_depth_path)
    check_and_make(front_mask_path)

    for i, obs in enumerate(demo):
        # left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        # left_shoulder_depth = utils.float_array_to_rgb_image(
        #     obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        # left_shoulder_mask = Image.fromarray(
        #     (obs.left_shoulder_mask * 255).astype(np.uint8))
        # right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        # right_shoulder_depth = utils.float_array_to_rgb_image(
        #     obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        # right_shoulder_mask = Image.fromarray(
        #     (obs.right_shoulder_mask * 255).astype(np.uint8))
        # overhead_rgb = Image.fromarray(obs.overhead_rgb)
        # overhead_depth = utils.float_array_to_rgb_image(
        #     obs.overhead_depth, scale_factor=DEPTH_SCALE)
        # overhead_mask = Image.fromarray(
        #     (obs.overhead_mask * 255).astype(np.uint8))
        # wrist_rgb = Image.fromarray(obs.wrist_rgb)
        # wrist_depth = utils.float_array_to_rgb_image(
        #     obs.wrist_depth, scale_factor=DEPTH_SCALE)
        # wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        # front_rgb = Image.fromarray(obs.front_rgb)
        # front_depth = utils.float_array_to_rgb_image(
        #     obs.front_depth, scale_factor=DEPTH_SCALE)
        # front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        # left_shoulder_rgb.save(
        #     os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        # left_shoulder_depth.save(
        #     os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        # left_shoulder_mask.save(
        #     os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        # right_shoulder_rgb.save(
        #     os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        # right_shoulder_depth.save(
        #     os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        # right_shoulder_mask.save(
        #     os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
        # overhead_rgb.save(
        #     os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        # overhead_depth.save(
        #     os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        # overhead_mask.save(
        #     os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
        # wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        # wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        # wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        # front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        # front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        # front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        if i >= 0:
            left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
            left_shoulder_depth = utils.float_array_to_rgb_image(
                obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
            left_shoulder_mask = Image.fromarray(
                (obs.left_shoulder_mask * 255).astype(np.uint8))
            right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
            right_shoulder_depth = utils.float_array_to_rgb_image(
                obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
            right_shoulder_mask = Image.fromarray(
                (obs.right_shoulder_mask * 255).astype(np.uint8))
            overhead_rgb = Image.fromarray(obs.overhead_rgb)
            overhead_depth = utils.float_array_to_rgb_image(
                obs.overhead_depth, scale_factor=DEPTH_SCALE)
            overhead_mask = Image.fromarray(
                (obs.overhead_mask * 255).astype(np.uint8))
            wrist_rgb = Image.fromarray(obs.wrist_rgb)
            wrist_depth = utils.float_array_to_rgb_image(
                obs.wrist_depth, scale_factor=DEPTH_SCALE)
            wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
            front_rgb = Image.fromarray(obs.front_rgb)
            front_depth = utils.float_array_to_rgb_image(
                obs.front_depth, scale_factor=DEPTH_SCALE)
            front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))


            # if i == 0:
            #     print(np.min(obs.front_depth), np.max(obs.front_depth))
            #     print(obs.misc["front_camera_near"], obs.misc["front_camera_far"])
            #     exit()




            left_shoulder_rgb.save(
                os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
            left_shoulder_depth.save(
                os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
            left_shoulder_mask.save(
                os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
            right_shoulder_rgb.save(
                os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
            right_shoulder_depth.save(
                os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
            right_shoulder_mask.save(
                os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
            overhead_rgb.save(
                os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
            overhead_depth.save(
                os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
            overhead_mask.save(
                os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
            wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
            wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
            wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
            front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
            front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
            front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

            # front_intrinsics_path = os.path.join(example_path, 'front_intrinsics', f"{i}.txt")
            # os.makedirs(os.path.join(example_path, 'front_intrinsics'), exist_ok=True)
            # np.savetxt(front_intrinsics_path, obs.misc['front_camera_intrinsics'])                  

            # front_extrinsics_path = os.path.join(example_path, 'front_extrinsics', f"{i}.txt")
            # os.makedirs(os.path.join(example_path, 'front_extrinsics'), exist_ok=True)
            # np.savetxt(front_extrinsics_path, obs.misc['front_camera_extrinsics'])

            # grasp_obj_pose_path = os.path.join(example_path, 'grasp_obj_pose', f"{i}.txt")
            # os.makedirs(os.path.join(example_path, 'grasp_obj_pose'), exist_ok=True)
            # np.savetxt(grasp_obj_pose_path, obs.misc["grasp_obj_pose"] )
            
            


        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    with open(os.path.join(example_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)

def save_scene_pc(demo, frame_idx, save_dir):

    check_and_make(save_dir)

    obs = demo[frame_idx]

    import open3d as o3d
    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(obs.front_point_cloud.reshape(-1, 3))
    pc1.colors = o3d.utility.Vector3dVector(obs.front_rgb.reshape(-1, 3) / 255.)
    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(obs.overhead_point_cloud.reshape(-1, 3))
    pc2.colors = o3d.utility.Vector3dVector(obs.overhead_rgb.reshape(-1, 3) / 255.)
    pc3 = o3d.geometry.PointCloud()
    pc3.points = o3d.utility.Vector3dVector(obs.left_shoulder_point_cloud.reshape(-1, 3))
    pc3.colors = o3d.utility.Vector3dVector(obs.left_shoulder_rgb.reshape(-1, 3) / 255.)
    pc4 = o3d.geometry.PointCloud()
    pc4.points = o3d.utility.Vector3dVector(obs.right_shoulder_point_cloud.reshape(-1, 3))
    pc4.colors = o3d.utility.Vector3dVector(obs.right_shoulder_rgb.reshape(-1, 3) / 255.)

    # combine pc
    pc = pc1 + pc2 + pc3 + pc4
    
    # Create bounding box:
    import itertools
    bounds = [[-0.6, 0.6], [-0.6, 0.6], [0., 2.]]           # set the bounds
    bounding_box_points = list(itertools.product(*bounds))  # create limit points
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points))    # create bounding box object
    pc = pc.crop(bounding_box)
    pc = pc.voxel_down_sample(voxel_size=0.001)
    o3d.io.write_point_cloud(os.path.join(save_dir, "scene.ply"), pc)

    # convert in target's frame
    points = np.asarray(pc.points)
    from utils.pose_utils import get_rel_pose, euler_from_quaternion, compute_rel_transform
    import utils.transform_utils as T
    grasp_obj_pose = obs.misc["grasp_obj_pose"]
    target_obj_pose = obs.misc["target_obj_pose"]
    points = np.vstack([
            compute_rel_transform(target_obj_pose[:3], T.quat2mat(target_obj_pose[3:]), p[:3], T.quat2mat(grasp_obj_pose[3:]))[0]
            for p in points
        ]
    )
    pc.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    o3d.io.write_point_cloud(os.path.join(save_dir, "scene_in_target.ply"), pc)

    # visualization
    # vis = o3d.visualization.VisualizerWithEditing()
    # vis.create_window()
    # vis.add_geometry(pc)
    # vis.run()  # user picks points
    # vis.destroy_window()

def run(i, lock, task_index, variation_count, results, file_lock, tasks, data_dict):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = MyEnvironmentPeract(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True)
    rlbench_env.launch()

    task_env = None

    tasks_with_problems = results[i] = ''

    # collect data for dp3 (all demo)
    cur_data_dict = data_dict[i] = {
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
    while True:
        # Figure out what task/variation this thread is going to do
        with lock:

            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break
            t = tasks[task_index.value]
        
        variation_path = os.path.join(
            FLAGS.save_path, f"{FLAGS.split}", task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count)
        check_and_make(variation_path)
        
        # save all variation description
        task_env = rlbench_env.get_task(t)
        possible_variations = task_env.variation_count()
        var_desc_dict = {}
        for var in range(possible_variations):
            task_env = rlbench_env.get_task(t)
            task_env.set_variation(var)
            descriptions, obs = task_env.reset()
            var_desc_dict[var] = descriptions

        with open(os.path.join(variation_path, "all_variation_descriptions.pkl"), 'wb') as f:
            pickle.dump(var_desc_dict, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(FLAGS.episodes_per_task):
            print('Process', i, '// Task:', task_env.get_name(),
                  '// Variation:', my_variation_count, '// Demo:', ex_idx)
            attempts = 10
            while attempts > 0:
                try:
                    task_env = rlbench_env.get_task(t)
                    task_env.set_variation(my_variation_count)
                    descriptions, obs = task_env.reset()
                    # print("scene_var", task_env._scene._variation_index)
                    # print("task_env_var", task_env._task._variation_number)
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)
                    # print("scene_var", task_env._scene._variation_index)
                    # print("task_env_var", task_env._task._variation_number)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), my_variation_count, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                with file_lock:
                    if ex_idx == 0:
                        frame_idx = 0
                        save_path = episode_path
                        save_scene_pc(demo, frame_idx, save_path)

                        
                    save_demo(demo, episode_path, my_variation_count)
                    episode_keypoints = keypoint_discovery(demo)


                    # euler
                    # grasp_obj_pose_list = [obs.misc["obj_pose_relative_to_target_euler"] for obs in demo]
                    # target_obj_pose_list = [obs.misc["target_pose_relative_to_target_euler"] for obs in demo]
                    # quaternion
                    poses_dict = {
                        "task_stage": [],
                        "grasp_obj_pose": [],
                        "grasp_obj_pose_relative_to_target": [],
                        "target_obj_pose": [],
                        "keypoint": [],
                    }
                    count = 0
                    # grasp_obj_pose_list = []
                    # grasp_obj_pose_relative_to_target_list = []
                    # target_obj_pose_list = []
                    stage_idx = 0
                    for idx, (gripper_open, obj_pose, obj_pose_relative_to_target, target_pose) in enumerate(zip(
                        [obs.gripper_open for obs in demo],
                        [obs.misc["grasp_obj_pose"] for obs in demo],
                        [obs.misc["obj_pose_relative_to_target"] for obs in demo],
                        [obs.misc["target_obj_pose"] for obs in demo])
                        ):
                        if gripper_open == 0: # gripper_close
                            # grasp_obj_pose_list.append(obj_pose)
                            # grasp_obj_pose_relative_to_target_list.append(obj_pose_relative_to_target)
                            # target_obj_pose_list.append(target_pose)
                            poses_dict["grasp_obj_pose"].append(obj_pose)
                            poses_dict["grasp_obj_pose_relative_to_target"].append(obj_pose_relative_to_target)
                            poses_dict["target_obj_pose"].append(target_pose)
                            poses_dict["task_stage"].append(stage_idx)
                            if idx in episode_keypoints:
                                poses_dict["keypoint"].append(count)
                            count += 1

                    # state_arrays_sub, state_in_world_arrays_sub, state_next_arrays_sub, goal_arrays_sub, action_arrays_sub, total_count_sub = collect_narr_function(grasp_obj_pose_list, grasp_obj_pose_relative_to_target_list, target_obj_pose_list, None)
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
                        # cprint('Episode: {}, Reward: {}, Success Times: {}'.format(episode_idx, ep_reward, ep_success_times), 'green')
                        # episode_idx += 1

                    print("%d state-action pairs collected." % len(cur_data_dict["action_arrays"]))
                    print(
                        "\tTask: %s // Demo Length: %d"
                        % (task_env.get_name(), len(demo))
                    )



                    with open(os.path.join(
                            episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                        pickle.dump(descriptions, f)
                break
            if abort_variation:
                break
        data_dict[i] = cur_data_dict

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def run_all_variations(i, lock, task_index, variation_count, results, file_lock, tasks, data_dict, peract_demo_dir=None):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = MyEnvironmentPeract(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True)
    # rlbench_env = Environment(
    #     action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
    #     obs_config=obs_config,
    #     headless=False)
    rlbench_env.launch()

    task_env = None

    tasks_with_problems = results[i] = ''    
    
    # collect data for dp3 (all demo)
    cur_data_dict = data_dict[i] = {
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
    
    while True:
        # with lock:
        if task_index.value >= num_tasks:
            print('Process', i, 'finished')
            break

        t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        possible_variations = task_env.variation_count()

        variation_path = os.path.join(
            FLAGS.save_path, f"{FLAGS.split}", task_env.get_name(),
            VARIATIONS_ALL_FOLDER)
        check_and_make(variation_path)
        
        # save all variation description
        task_env = rlbench_env.get_task(t)
        possible_variations = task_env.variation_count()
        var_desc_dict = {}
        for var in range(possible_variations):
            task_env = rlbench_env.get_task(t)
            task_env.set_variation(var)
            descriptions, obs = task_env.reset()
            var_desc_dict[var] = descriptions
            
        with open(os.path.join(variation_path, "all_variation_descriptions.pkl"), 'wb') as f:
            pickle.dump(var_desc_dict, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(FLAGS.episodes_per_task):
            attempts = 10
            while attempts > 0:
                try:
                    task_env = rlbench_env.get_task(t)
                    if peract_demo_dir is None:
                        # random generated demo
                        variation = np.random.randint(possible_variations)
                        task_env.set_variation(variation)
                        descriptions, obs = task_env.reset()
                    else:
                        # load pre-grnerated demo from preact
                        peract_demo_path = os.path.join(peract_demo_dir, f"{FLAGS.split}", task_env.get_name(), "all_variations", "episodes", f"episode{ex_idx}")
                        print(f"load preact's demo from the path {peract_demo_path}")
                        with open(os.path.join(peract_demo_path, LOW_DIM_PICKLE), "rb") as fin:
                            peract_demo = pickle.load(fin)
                        with open(os.path.join(peract_demo_path, VARIATION_NUMBER), 'rb') as f:
                            variation = pickle.load(f)
                        task_env.set_variation(variation)
                        descriptions, obs = task_env.reset_to_demo(peract_demo)

                    print('Process', i, '// Task:', task_env.get_name(),
                            '// Variation:', variation, '// Demo:', ex_idx)

                    # print("scene_var", task_env._scene._variation_index)
                    # print("task_env_var", task_env._variation_number)
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)
                    # print("scene_var", task_env._scene._variation_index)
                    # print("task_env_var", task_env._variation_number)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), variation, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                with file_lock:
                    if ex_idx == 0:
                        frame_idx = 0
                        save_path = episode_path
                        save_scene_pc(demo, frame_idx, save_path)
                    


                    save_demo(demo, episode_path, variation)
                    episode_keypoints = keypoint_discovery(demo)



                    # quaternion
                    poses_dict = {
                        "task_stage": [],
                        "grasp_obj_pose": [],
                        "grasp_obj_pose_relative_to_target": [],
                        "target_obj_pose": [],
                        "keypoint": [],
                    }
                    count = 0
                    stage_idx = 0
                    for idx, (gripper_open, obj_pose, obj_pose_relative_to_target, target_pose) in enumerate(zip(
                        [obs.gripper_open for obs in demo],
                        [obs.misc["grasp_obj_pose"] for obs in demo],
                        [obs.misc["obj_pose_relative_to_target"] for obs in demo],
                        [obs.misc["target_obj_pose"] for obs in demo])
                        ):
                        if gripper_open == 0: # gripper_close
                            poses_dict["grasp_obj_pose"].append(obj_pose)
                            poses_dict["grasp_obj_pose_relative_to_target"].append(obj_pose_relative_to_target)
                            poses_dict["target_obj_pose"].append(target_pose)
                            poses_dict["task_stage"].append(stage_idx)
                            if idx in episode_keypoints:
                                poses_dict["keypoint"].append(count)
                            count += 1

                    
                    if len(poses_dict["grasp_obj_pose"]) <= 0:
                        print(f"Skip episode{ex_idx}. Gripper always open.")
                        break

                    arrays_sub_dict_list, total_count_sub_list = collect_narr_function([poses_dict], None)
                    assert len(arrays_sub_dict_list) == len(total_count_sub_list) == 1
                    
                    for arrays_sub_dict, total_count_sub in zip(arrays_sub_dict_list, total_count_sub_list):
                        cur_data_dict["total_count"] += total_count_sub
                        cur_data_dict["episode_ends_arrays"].append(copy.deepcopy(cur_data_dict["total_count"])) # the index of the last step of the episode    
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

                    print("%d state-action pairs collected." % len(cur_data_dict["action_arrays"]))
                    print(
                        "\tTask: %s // Demo Length: %d"
                        % (task_env.get_name(), len(demo))
                    )



                    with open(os.path.join(
                            episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                        pickle.dump(descriptions, f)
                break
            if abort_variation:
                break
        data_dict[i] = cur_data_dict

        # with lock:
        task_index.value += 1

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def main(argv):
    assert len(FLAGS.tasks) == 1, f"Only support one task for now (save zarr for each task)"
    assert FLAGS.split in ['train', 'val', 'test']
    if FLAGS.split in ['train']:
        assert FLAGS.episodes_per_task <= 100
    else:
        assert FLAGS.episodes_per_task <= 25
    if not FLAGS.all_variations:
        assert FLAGS.variations == 1, f"only support one variations if not FLAGS.all_variations (zarr is saved for each variation)"


    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    data_dict = manager.dict()
    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    if FLAGS.all_variations:
        # multiprocessing for when all_variations=True is not supported
        peract_demo_dir = FLAGS.peract_demo_dir
        run_all_variations(0, lock, task_index, variation_count, result_dict, file_lock, tasks, data_dict, 
                           peract_demo_dir=peract_demo_dir)
    else:
        processes = [Process(
            target=run, args=(
                i, lock, task_index, variation_count, result_dict, file_lock,
                tasks, data_dict))
            for i in range(FLAGS.processes)]
        [t.start() for t in processes]
        [t.join() for t in processes]

    print('Data collection done!')
    # for i in range(FLAGS.processes):
    #     print(result_dict[i])


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
    for i in range(FLAGS.processes):
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
    # !! assume only one task
    if FLAGS.all_variations:
        save_dir = os.path.join(FLAGS.save_path, f"{FLAGS.split}", FLAGS.tasks[0], VARIATIONS_ALL_FOLDER, "zarr")
    else:
        # TODO: support multiple variations
        cur_variation = 0
        save_dir = os.path.join(FLAGS.save_path, f"{FLAGS.split}", FLAGS.tasks[0], VARIATIONS_FOLDER % cur_variation, "zarr")
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
if __name__ == '__main__':
  app.run(main)