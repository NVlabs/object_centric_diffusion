from typing import Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
# import pytorch3d.ops as torch3d_ops

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d_progress import ConditionalUnet1D_progress
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder
from utils.vis_utils import dp3_visualize


def custom_normalize(data, normalizer, key):
    pose = copy.deepcopy(data) # (B, T, 7)
    if key is not None:
        use_progress = True if pose[key].shape[-1] > 7 else False
    else:
        use_progress = True if pose.shape[-1] > 7 else False

    if key is not None:
        device = pose[key].device
        # save and normalize quaternion
        if use_progress:
            progress = pose[key][:, :, 7:8]
        pose_ori = pose[key][:, :, 3:7] 
        # print("pose[key]", pose[key].shape)
        # remove quaternion
        pose[key] = pose[key][:, :, :3]
    else:
        device = pose.device
        # save and normalize quaternion
        if use_progress:
            progress = pose[:, :, 7:8] 
        pose_ori = pose[:, :, 3:7] 
        # print("pose", pose.shape)
        # remove quaternion
        pose = pose[:, :, :3]

    # normalize data and quaternion
    pose_ori = torch.nn.functional.normalize(pose_ori, dim=2)
    npose = normalizer.normalize(pose)

    if key is not None:
        pose_ori = pose_ori.to(npose[key].device)
        # add the normalize quaternion back
        if use_progress:
            progress = progress.to(npose[key].device)
            npose[key] = torch.concat([npose[key], pose_ori, progress], dim=-1)
        else:
            npose[key] = torch.concat([npose[key], pose_ori], dim=-1)
        # print("data[key]", data[key].shape)
        # print("npose[key]", npose[key].shape)
    else:
        pose_ori = pose_ori.to(npose.device)
        if use_progress:
            progress = progress.to(npose.device)
            npose = torch.concat([npose, pose_ori, progress], dim=-1)
        else:
            npose = torch.concat([npose, pose_ori], dim=-1)
        # print("data", data.shape)
        # print("npose", npose.shape)
    return npose


def custom_unnormalize(data, normalizer, key):
    pose = copy.deepcopy(data) # (B, T, 7)
    if key is not None:
        use_progress = True if pose[key].shape[-1] > 7 else False
    else:
        use_progress = True if pose.shape[-1] > 7 else False

    if key is not None:
        # save and normalize quaternion
        if use_progress:
            progress = pose[key][:, :, 7:8]
        pose_ori = pose[key][:, :, 3:7] 
        # print("pose[key]", pose[key].shape)
        # remove quaternion
        pose[key] = pose[key][:, :, :3]
    else:
        # save and normalize quaternion
        if use_progress:
            progress = pose[:, :, 7:8] 
        pose_ori = pose[:, :, 3:7] 
        # print("pose", pose.shape)
        # remove quaternion
        pose = pose[:, :, :3]

    # normalize data and quaternion
    pose_ori = torch.nn.functional.normalize(pose_ori, dim=2)
    npose = normalizer.unnormalize(pose)

    if key is not None:
        # add the normalize quaternion back
        if use_progress:
            npose[key] = torch.concat([npose[key], pose_ori, progress], dim=-1)
        else:
            npose[key] = torch.concat([npose[key], pose_ori], dim=-1)
        # print("npose[key]", npose[key].shape)
    else:
        if use_progress:
            npose = torch.concat([npose, pose_ori, progress], dim=-1)
        else:
            npose = torch.concat([npose, pose_ori], dim=-1)
        # print("npose", npose.shape)
    return npose


class SimpleDP3(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            use_lang_emb=False,
            use_stage_emb=False,
            use_progress=False,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            predict_type = "relative",
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        self.use_progress = use_progress
        if use_progress:
            assert len(action_shape) == 1 and action_shape[0] == 8
        else:
            assert len(action_shape) == 1 and action_shape[0] == 7
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                 img_crop_shape=crop_shape,
                                 out_channel=encoder_output_dim,
                                 pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                 use_pc_color=use_pc_color,
                                 pointnet_type=pointnet_type,
                                 use_lang_emb=use_lang_emb,
                                 use_stage_emb=use_stage_emb,
                                )
        cprint(f"[SDP3] use_lang_emb: {use_lang_emb}", "yellow")
        cprint(f"[SDP3] use_stage_emb: {use_stage_emb}", "yellow")

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[SDP3] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[SDP3] pointnet_type: {self.pointnet_type}", "yellow")

        if use_progress:
            input_dim = input_dim - 1 # handle progress/gripper separately
            model = ConditionalUnet1D_progress(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                condition_type=condition_type,
                use_down_condition=use_down_condition,
                use_mid_condition=use_mid_condition,
                use_up_condition=use_up_condition,
            )
        else:
            model = ConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                condition_type=condition_type,
                use_down_condition=use_down_condition,
                use_mid_condition=use_mid_condition,
                use_up_condition=use_up_condition,
            )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.predict_type = predict_type

        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], target=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        
        # (debug only) check if they are in the same frame
        # idx = 10
        # states = obs_dict['agent_pos']
        # actions = target
        # prev_state = states[idx-1] # previous pose
        # current_state = states[idx] # current pose
        # current_state2 = prev_state + actions[idx-1]
        # print(current_state, current_state2) # should be the same

        # normalize input
        # nobs = self.normalizer.normalize(obs_dict)

        # normalize quaternion seperately
        nobs = custom_normalize(obs_dict, self.normalizer, key='agent_pos')

        if 'point_cloud' in nobs:
            # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
            if not self.use_pc_color:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))

        # !! Important
        if value.shape[1] >1 :
            value = value[:, 0].unsqueeze(1)

        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        # action_pred = self.normalizer['action'].unnormalize(naction_pred)
        action_pred = custom_unnormalize(naction_pred, self.normalizer['action'], key=None)
        # naction_pred2 = custom_normalize(action_pred, self.normalizer['action'], key=None)
        # from env_rlbench.utils.pose_utils import get_rel_pose, euler_from_quaternion
        # print(naction_pred[0], naction_pred2[0])
        # print(euler_from_quaternion(*naction_pred[0][0][3:]), euler_from_quaternion(*naction_pred2[0][0][3:]))
        # exit()
        
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        # get prediction
        result = {
            'action': action,           # [B, 3 ,7]
            'action_pred': action_pred, # [B, 4 ,7]
        }


        # (debug only) check if they are in the same frame
        # idx = 10
        # states = self.normalizer['agent_pos'].unnormalize(nobs['agent_pos'])
        # actions = target #self.normalizer['action'].unnormalize(target)
        # prev_state = states[idx-1] # previous pose
        # current_state = states[idx] # current pose
        # current_state2 = prev_state.cpu() + actions[idx-1].cpu()
        # print(current_state, current_state2) # should be the same
        # exit()


        # visualize and loss (debug only)
        visualize = False
        predict_type = 'relative' #'absolute'
        if visualize:
            dp3_visualize(
                # self.normalizer['agent_pos'].unnormalize(nobs['agent_pos']), 
                custom_unnormalize(nobs['agent_pos'], self.normalizer['agent_pos'], key=None),
                action_pred, 
                target, # target is not normalized
                predict_type=predict_type,
            )
        if target is not None:
            action_pred = action_pred.to(target.device)
            # loss = F.mse_loss(action_pred, target, reduction='none')
            # loss = reduce(loss, 'b ... -> b (...)', 'mean')
            # loss = loss.mean()
            # result["loss"] = loss.item()

            mse = torch.nn.functional.mse_loss(action_pred, target)
            result["loss"] = mse.item()
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        # nobs = self.normalizer.normalize(batch['obs'])
        # nactions = self.normalizer['action'].normalize(batch['action'])

        # normalize quaternion seperately        
        nobs = custom_normalize(batch['obs'], self.normalizer, key='agent_pos')
        nactions = custom_normalize(batch['action'], self.normalizer['action'], key=None)

        if 'point_cloud' in nobs:
            if not self.use_pc_color:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
       
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            if 'point_cloud' in nobs:
                # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
                this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
                this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        


        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")


        visualize = False
        predict_type = 'relative'
        if visualize:
            dp3_visualize(
                # self.normalizer['agent_pos'].unnormalize(nobs['agent_pos']), 
                # self.normalizer['action'].unnormalize(pred), 
                # self.normalizer['action'].unnormalize(target)
                custom_unnormalize(nobs['agent_pos'], self.normalizer['agent_pos'], key=None),
                # pred=custom_unnormalize(pred.clone().detach(), self.normalizer['action'], key=None),
                target=custom_unnormalize(target, self.normalizer['action'], key=None),
                predict_type=predict_type,
            )

        # (debug only) check if they are in the same frame
        # idx = 10
        # states = self.normalizer['agent_pos'].unnormalize(nobs['agent_pos'])
        # actions = self.normalizer['action'].unnormalize(target)
        # prev_state = states[idx-1] # previous pose
        # current_state = states[idx] # current pose
        # current_state2 = prev_state.cpu() + actions[idx-1].cpu()
        # print(current_state, current_state2) # should be the same
        # exit()

        if self.use_progress:
            # continous loss for progress
            # loss = F.mse_loss(pred, target, reduction='none')
            # loss[..., -1:] *= 0.1

            # binary loss for gripper
            loss_trans = F.mse_loss(pred[..., :3], target[..., :3], reduction='none')

            loss_ori = F.mse_loss(pred[..., 3:-1], target[..., 3:-1], reduction='none')
            # loss_ori2 = F.mse_loss(pred[..., 3:-1], -target[..., 3:-1], reduction='none')
            # loss_ori[loss_ori2 < loss_ori] = loss_ori2[loss_ori2 < loss_ori]    # symmetric_rotation_loss

            loss_gripper = F.binary_cross_entropy(pred[..., -1:], target[..., -1:], reduction='none')
            # loss_gripper = F.mse_loss(pred[..., -1:], target[..., -1:], reduction='none')

            loss = torch.concat([loss_trans, loss_ori, loss_gripper], dim=-1)
            loss[..., -1:] *= 0.1
        else:
            loss = F.mse_loss(pred, target, reduction='none')

        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()


        # import einops
        # from diffusion_policy_3d.model.common.geodesic_loss import GeodesicLoss, quaternion_to_matrix
        # pred_rot = quaternion_to_matrix(pred[:, :, 3:])
        # target_rot = quaternion_to_matrix(target[:, :, 3:])
        # geodesic_loss = []
        # for cur_pred_rot, cur_target_rot in zip(pred_rot, target_rot):
        #     geodesic_loss.append(
        #             GeodesicLoss(reduction='none')(
        #             cur_pred_rot,
        #             cur_target_rot,
        #         )
        #     )
        # geodesic_loss = torch.stack(geodesic_loss).unsqueeze(-1)
        # geodesic_loss = einops.repeat(geodesic_loss, 'b h 1 -> b h d', d=4) # (b, h, 4)
        # mse_loss = F.mse_loss(pred[:, :, :3], target[:, :, :3], reduction='none')
        # loss = torch.concat([mse_loss, geodesic_loss], dim=-1)
        # loss = loss * loss_mask.type(loss.dtype)
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss.mean()



        loss_dict = {
                # 'mse_loss': reduce(mse_loss, 'b ... -> b (...)', 'mean').mean(),
                # 'geodesic_loss': reduce(geodesic_loss, 'b ... -> b (...)', 'mean').mean(),
                'bc_loss': loss.item(),
            }

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
        
        return loss, loss_dict