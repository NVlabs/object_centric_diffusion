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

import numpy as np
import torch
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.env_runner.base_runner import BaseRunner

import glob
import os
import pickle
import numpy as np
from termcolor import cprint

from pyrep.const import RenderMode
from pyrep.objects.shape import Shape
from rlbench.backend import utils
from rlbench.backend.const import *
from rlbench.action_modes.arm_action_modes import (
    ArmActionMode,
    EndEffectorPoseViaIK,
    EndEffectorPoseViaPlanning,
    assert_action_shape,
)
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from rlbench.backend import task as rlbench_tasks
from env_rlbench.policy.dp3_policy import RLBenchDP3Policy
from env_rlbench.policy.subgoal_policy import RLBenchSubGoalPolicy
from env_rlbench.runner.rl_bench_dataset import _create_obs_config, _get_action_mode
from env_rlbench.runner.rl_bench_env import CustomRLBenchEnv

from diffusion_policy_3d.model.common.geodesic_loss import GeodesicLoss
from diffusion_policy_3d.model.clip.clip import build_model, load_clip, tokenize
from utils.pose_utils import euler_from_quaternion


class RLBenchRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 root_dir,
                 task_name,
                 has_lang_emb,
                 has_stage_emb,
                 enable_stage,
                 use_fp,
                 fp_cam_name,
                 pose_estimation_wrapper,
        ):
        super().__init__(output_dir)

        # ------------------------
        # |    Pose Estimation   |
        # ------------------------
        self.pose_estimation_wrapper = pose_estimation_wrapper

        # ------------------------
        # |        RLBench       |
        # ------------------------

        # create env
        self.task = task_name
        self.enable_stage = enable_stage
        self.env = self._create_env(task=self.task, root_dir=root_dir, enable_stage=enable_stage, use_fp=use_fp, fp_cam_name=fp_cam_name, pose_estimation_wrapper=self.pose_estimation_wrapper, headless=True)
        self.env.launch()
        self.use_fp = use_fp
        
        # load demo for evalution
        self.root_dir = root_dir
        self.demo_path_list = glob.glob(os.path.join(self.root_dir, "episodes", "episode*", "low_dim_obs.pkl"))
        assert len(self.demo_path_list) > 0
        self.demo_path_list.sort()

        # language embeding
        self.has_lang_emb = has_lang_emb
        # if self.has_lang_emb:
        self._lang_token_embs = self.get_language_embedding()
        cprint(f"[RLbench Runner] has_lang_emb: {has_lang_emb}", "yellow")

        self.has_stage_emb = has_stage_emb
        cprint(f"[RLbench Runner] has_stage_emb: {has_stage_emb}", "yellow")


    def _create_env(self, task, root_dir, enable_stage=False, use_fp=False, fp_cam_name=None, pose_estimation_wrapper=None, headless=True):
        _camera_names = ["front", "left_shoulder", "right_shoulder", "wrist", "overhead"]
        _ds_img_size = 128
        _data_raw_path = root_dir # !! This path does not do anything, but it needs to exist to prevent errors...
        
        observation_config = _create_obs_config(
            _camera_names,
            [_ds_img_size, _ds_img_size],
        )

        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
            # arm_action_mode=EndEffectorPoseViaIK(collision_checking=False),
            gripper_action_mode=Discrete()
        )

        task_files = [
            t.replace(".py", "")
            for t in os.listdir(rlbench_tasks.TASKS_PATH)
            if t != "__init__.py" and t.endswith(".py")
        ]
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_class = task_file_to_task_class(task)

        return CustomRLBenchEnv(
            task_class=task_class,
            observation_config=observation_config,
            action_mode=action_mode,
            episode_length=200,
            dataset_root=_data_raw_path,
            headless=headless,
            time_in_state=True,
            enable_stage=enable_stage,
            use_fp=use_fp,
            fp_cam_name=fp_cam_name,
            pose_estimation_wrapper=pose_estimation_wrapper,
        )

    def get_language_embedding(self):
        # load clip model
        model, _ = load_clip("RN50", jit=False, device='cuda')
        clip_model = build_model(model.state_dict())
        clip_model.to('cuda')
        del model

        # load all descriptions                
        description_path = os.path.join(self.root_dir, "all_variation_descriptions.pkl")
        print(f"Loading demo from the path {description_path}")
        with open(description_path, "rb") as fin:
            descriptions = pickle.load(fin) # dict of VAR: DESC

        cur_task_id = 0 # cur_task_id is always zero as there is only one task

        # pre-generate all language embeddings
        _lang_token_embs = {}
        n_variation = len(descriptions)
        for var_id in range(n_variation):
            cur_description = descriptions[var_id]

            tokens = tokenize(cur_description).numpy()
            token_tensor = torch.from_numpy(tokens).to('cuda')
            sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)

            # print(cur_description)      # 5 sentence
            # print(token_embs.shape)     # [5, 77, 512]
            # print(sentence_emb.shape)   # [5, 1024]

            lang_token_embs = sentence_emb[0].float().detach().cpu().numpy()[None, :]

            _lang_token_embs[f"{cur_task_id}_{var_id}"] = lang_token_embs
        return _lang_token_embs
    
    def get_observation(self, obs=None):
        if obs is None:
            obs = self.env.env._scene.get_observation()
        return obs

    def run(self, policy: BasePolicy=None, tag="latest", save_video=False):

        # get policy
        if policy is None: 
            subgoal_policy = RLBenchSubGoalPolicy(self.env)
        else:
            subgoal_policy = RLBenchDP3Policy(self.env, policy, self.use_fp, self.enable_stage)
            # subgoal_policy = RLBenchSubGoalPolicy(self.env)
        self.get_action = subgoal_policy.get_action
    

        success_list = []
        for demo_idx, demo_path in enumerate(self.demo_path_list):

            RUN_FINISHED = False
            N_TRIAL = 3
            for cur_trial in range(N_TRIAL):
                try:
                    # load demo
                    print(f"{demo_idx+1}/{len(self.demo_path_list)} Loading demo from the path {demo_path} (trial: {cur_trial})")
                    with open(demo_path, "rb") as fin:
                        demo = pickle.load(fin)
                    with open(demo_path.replace("low_dim_obs.pkl", "variation_number.pkl"), 'rb') as f:
                        variation_number = pickle.load(f)

                    # load language embeddings
                    lang_token_embs = None
                    cur_task_id = 0 # cur_task_id is always zero as there is only one task
                    if self.has_lang_emb:
                        if self.has_stage_emb:
                            var_id = 0  # train with stage embedding (i.e., train with only one variant / ignore task description)
                        else:
                            var_id = variation_number
                        lang_token_embs = self._lang_token_embs[f"{cur_task_id}_{var_id}"]
                    else:
                        lang_token_embs = self._lang_token_embs[f"{0}_{0}"]

                    # reset environment        
                    self.env.reset_to_demo(demo)
                    self.env.set_variation(variation_number)

                    # !! set variation_index and reset again (no idea why we need this...)
                    self.env._rlbench_env._scene._variation_index = variation_number
                    self.env.reset_to_demo(demo)

                    print("variation_number", variation_number)

                    # reset policy
                    subgoal_policy.reset()

                    # reset loop number
                    subgoal_policy.cur_loop_num = 0
                    if self.enable_stage:
                        if self.task == "place_cups":
                            subgoal_policy.max_loop_num = variation_number+1
                        elif self.task == "stack_cups":
                            subgoal_policy.max_loop_num = 2
                        elif self.task == "stack_blocks":
                            subgoal_policy.max_loop_num = self.env.env._scene.task.blocks_to_stack
                        else:
                            raise NotImplementedError
                    else:
                        subgoal_policy.max_loop_num = 1
                        
                    # reset parameters
                    success = False
                    term = False

                    # set episode length
                    if self.task == "place_cups" or self.task == "stack_blocks":
                        episode_length = 400 + 10
                    elif self.task == "stack_cups":
                        episode_length = 100 + 10
                    elif self.task == "light_bulb_in":
                        episode_length = 50 #25 + 10
                    elif self.task == "reach_and_drag":
                        episode_length = 50 + 10
                    else:
                        episode_length = 50 + 10

                    # initialize observation
                    obs = self.get_observation()

                    # start running
                    for step_i in range(episode_length):
                        # load stage embeddings
                        stage_embs = None
                        if self.has_stage_emb:
                            stage_id = subgoal_policy.cur_loop_num
                            stage_embs = np.zeros((1, 3)).astype(np.float32)
                            stage_embs[0][stage_id] = 1.
                        else:
                            stage_embs = np.zeros((1, 3)).astype(np.float32)
                        
                        # get action
                        if step_i < episode_length - 10:
                            action = self.get_action(obs, lang_token_embs=lang_token_embs, stage_embs=stage_embs)
                        else:
                            action = subgoal_policy._open_gripper()
                        
                        # apply action
                        if action is None:
                            obs = self.get_observation()
                            self.env._rlbench_env._scene.step()
                            self.env.env._scene.task.step()
                        else:                    
                            # Update the observation based on the predicted action
                            ts = self.env.step(action, record=False, verbose=True)
                            obs = self.get_observation()
                            term = ts.terminal

                        success, _ = self.env.env._scene.task.success()

                        # end condition
                        if success:
                            cprint(f"[RLbench Runner] Task success (in {step_i} steps).", "green")
                            RUN_FINISHED = True
                            break # break from execution loop
                        if term:
                            cprint(f"[RLbench Runner] Task fails. Error occurs.", "red")
                            RUN_FINISHED = True
                            break # break from execution loop
                        if episode_length > -1 and step_i >= episode_length-1:
                            cprint(f"Task fails. Exceed maximum episode length ({episode_length}).", "red")
                            RUN_FINISHED = True
                            break # break from execution loop
                except Exception as e:
                    print(e)
                if success:
                    break   # break from trial loop


            if RUN_FINISHED:
                # log result
                success_list.append(success)
                
                if self.use_fp:
                    save_path = os.path.join(self.output_dir, f"eval_use_fp=true_{self.task}")
                else:
                    save_path = os.path.join(self.output_dir, f"eval_use_fp=false_{self.task}")

                os.makedirs(save_path, exist_ok=True)


                # save result (per epoch)
                log_data_temp = {
                    "success": np.array(success_list),
                    "mean_success_rates": np.mean(success_list),
                }
                json_path = os.path.join(save_path, f'{tag}_temp.json')
                from utils.io_utils import save_np_dict_to_json
                save_np_dict_to_json(log_data_temp, json_path)

                # save logger data
                env_logger = self.env.get_env_logger()

                # save gif
                # gif_path = os.path.join(save_path, f"{tag}_{self.task}_{demo_idx}.gif")
                # if self.use_fp:
                #     env_logger.save_data(["vis_pose"], gif_path, type="gif")
                # else:
                #     env_logger.save_data(["obs_record_rgb"], gif_path, type="gif")

                # save video
                video_path = os.path.join(save_path, f"{tag}_{self.task}_{demo_idx}.mp4")
                if self.use_fp:
                    env_logger.save_data(["vis_pose"], video_path, type="video")
                    # env_logger.save_data(["obs_record_rgb"], video_path, type="video")
                else:
                    env_logger.save_data(["obs_record_rgb"], video_path, type="video")


                # get poses
                poses_est = env_logger.get_data("pose_est")
                poses_gt = env_logger.get_data("pose_gt")

                import utils.transform_utils as T
                def geodesic(quat1, quat2):
                    mat1 = torch.from_numpy(T.quat2mat(quat1)).unsqueeze(0)
                    mat2 = torch.from_numpy(T.quat2mat(quat1)).unsqueeze(0)
                    loss_fn = GeodesicLoss()
                    loss = loss_fn(mat1, mat2)
                    return loss.clone().cpu().numpy()
                def euler_dist(quat1, quat2):
                    euler1 = euler_from_quaternion(*quat1)
                    euler2 = euler_from_quaternion(*quat2)
                    return euler2-euler1

                # save pose diff
                import matplotlib.pyplot as plt
                import seaborn as sns

                pose_img_path = os.path.join(save_path, f"{tag}_{self.task}_{demo_idx}_pose_diff.png")
                trans_diff = [np.mean(np.abs(p1[:3] - p2[:3])) for p1, p2 in zip(poses_est, poses_gt)]
                # rot_diff = [np.mean(np.abs(euler_dist(p1[3:], p2[3:]))) for p1, p2 in zip(poses_est, poses_gt)]
                rot_diff = [np.mean(np.abs(geodesic(p1[3:], p2[3:]))) for p1, p2 in zip(poses_est, poses_gt)]
                
                fig, (ax1, ax2) = plt.subplots(2, 1)
                sns.lineplot(x=range(len(trans_diff)), y=trans_diff, ax=ax1)
                sns.lineplot(x=range(len(rot_diff)), y=rot_diff, ax=ax2)
                ax1.set_ylabel("Translation Error")
                ax2.set_ylabel("Rotation Error")
                plt.tight_layout()
                plt.savefig(pose_img_path)
                # plt.show()

        # convert every item to np array
        log_data = {
            "success": np.array(success_list),
            "mean_success_rates": np.mean(success_list),
        }
        return log_data