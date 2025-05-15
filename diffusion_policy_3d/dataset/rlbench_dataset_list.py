import yaml
import numpy as np
import torch
import copy
from typing import Dict
import hydra

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


class RLBenchDatasetList(BaseDataset):
    def __init__(self, root_dir=None):
        self.task_list = [
            'meat_off_grill',
            'put_money_in_safe',
            'place_wine_at_rack_location',
            'reach_and_drag',
            'stack_blocks',
            'close_jar',
            'light_bulb_in',
            'put_groceries_in_cupboard',
            'place_shape_in_shape_sorter',
            'insert_onto_square_peg',
            'stack_cups',
            'place_cups',
            'turn_tap',
        ]
        self.task_id_to_name = {}
        self.task_name_to_dataset = {}

        self.dataset_list = []
        self.global_idx_to_task_id_local_idx = {}

        start = 0
        for task_idx, task_name in enumerate(self.task_list):

            # read yaml
            with open(f"config/task/rlbench/{task_name}.yaml", 'r') as stream:
                dataset_cfg = yaml.safe_load(stream)["dataset"]

            if root_dir is not None:
                dataset_cfg["root_dir"] = f"{root_dir}/{task_name}/all_variations"  # overrides all single-task configs
            else:
                dataset_cfg["root_dir"] = dataset_cfg["root_dir"].replace("${task.task_name}", task_name)

            dataset = hydra.utils.instantiate(dataset_cfg)
            self.dataset_list.append(dataset)

            self.task_id_to_name[task_idx] = task_name
            self.task_name_to_dataset[task_name] = dataset

            # global index to (task_id, local index)
            dataset_length = len(dataset)
            for sample_idx in range(dataset_length):
                self.global_idx_to_task_id_local_idx[start+sample_idx] = (task_idx, sample_idx)
            start += dataset_length

    def get_validation_dataset(self):
        val_set_list = copy.copy(self)
        return val_set_list
    
    def get_normalizer(self, mode='limits', **kwargs):
        action_all = []
        agent_pos_all = []
        for dataset in self.dataset_list:
            print(dataset.replay_buffer['action'][:, :3].shape)
            action_all.append(dataset.replay_buffer['action'][:, :3])
            agent_pos_all.append(dataset.replay_buffer['state'][...,:][:, :3])
        
        print(len(action_all))

        action_all = np.concatenate(action_all, axis=0)
        agent_pos_all = np.concatenate(agent_pos_all, axis=0)
        
        data = {
            'action': action_all,
            'agent_pos': agent_pos_all,
            # 'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    
    def __len__(self) -> int:
        return len(self.global_idx_to_task_id_local_idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_idx, sample_idx = self.global_idx_to_task_id_local_idx[idx]

        select_dataset = self.dataset_list[task_idx]
        return select_dataset.__getitem__(sample_idx)