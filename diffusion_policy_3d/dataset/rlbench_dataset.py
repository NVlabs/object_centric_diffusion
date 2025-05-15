from diffusion_policy_3d.dataset.rlbench_base_dataset import RLBenchBaseDataset


class RLBenchDataset(RLBenchBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)