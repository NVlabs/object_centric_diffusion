# SPOT: SE(3) Pose Trajectory Diffusion for Object-Centric Manipulation

[Cheng-Chun Hsu](https://chengchunhsu.github.io/), [Bowen Wen](https://research.nvidia.com/person/bowen-wen), [Jie Xu](https://research.nvidia.com/person/jie-xu), [Yashraj Narang](https://research.nvidia.com/person/yashraj-narang), [Xiaolong Wang](https://research.nvidia.com/labs/lpr/author/xiaolong-wang/), [Yuke Zhu](https://research.nvidia.com/person/yuke-zhu), [Joydeep Biswas](https://www.joydeepb.com/), [Stan Birchfield](https://research.nvidia.com/person/stan-birchfield)

ICRA 2025

[Project](https://nvlabs.github.io/object_centric_diffusion/) | [Paper](https://arxiv.org/abs/2411.00965)



## Abstract

We introduce SPOT, an object-centric imitation learning framework. The key idea is to capture each task by an object-centric representation, specifically the SE(3) object pose trajectory relative to the target. This approach decouples embodiment actions from sensory inputs, facilitating learning from various demonstration types, including both action-based and action-less human hand demonstrations, as well as cross-embodiment generalization.  


Additionally, object pose trajectories inherently capture planning constraints from demonstrations without the need for manually-crafted rules. 
To guide the robot in executing the task, the object trajectory is used to condition a diffusion policy. We systematically evaluate our method on simulation and real-world tasks. In real-world evaluation, using only eight demonstrations shot on an iPhone, our approach completed all tasks while fully complying with task constraints.



## Installation

The codebase is thoroughly tested on a desktop running Ubuntu 22 with an RTX 4090 GPU.



### Environment Setup
Create conda environment

```
conda create -n spot python=3.8
conda activate spot
```

Install dependencies (for FoundationPose compilation)

```
# Install eigen library
conda install conda-forge::eigen=3.4.0

# Install gcc and cuda
conda install gcc_linux-64 gxx_linux-64
conda install cuda -c nvidia/label/cuda-12.1.0
conda install nvidia/label/cuda-12.1.0::cuda-cudart
conda install cmake

# Install boost library
sudo apt install libboost-all-dev
conda install conda-forge::boost
```

Install Pytorch and Pytorch3d

```
conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytorch3d -c pytorch3d
```



### RLBench Installation

Install CoppeliaSim v4.1.0 (see [here](https://github.com/stepjam/PyRep#install) for details)

```
# set env variables
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

Install PyRep, YARR, and RLBench (PerAct's branch)
```
git clone https://github.com/MohitShridhar/PyRep.git
cd PyRep
pip3 install -r requirements.txt
pip3 install .
cd ..

git clone https://github.com/stepjam/YARR.git
cd YARR
pip3 install -r requirements.txt
pip3 install .
cd ..

git clone https://github.com/MohitShridhar/RLBench.git -b peract
cd RLBench
pip3 install -r requirements.txt
pip3 install .
cd ..
```



### FoundationPose and Dependencies Setup

Install python depdencies

```
pip install -r fp_requirements.txt
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

pip install -r dp3_requirements.txt
```

Compile  FoundationPose's extensions
```
cd foundation_pose
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.8/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
cd ..
```

Download model weight from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) or [Foundation Pose repo](https://github.com/NVlabs/FoundationPose?tab=readme-ov-file), and put the model weight under `data/model_weight/foundation_pose`.



## Training and Evaluation on RLBench

Our dataset generation is based on PerAct's pre-generated datasets. We replay the demonstrations to collect object pose information for policy training.



### Requirements

Download [PerAct's  pre-generated datasets](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ) for train (100 episodes), validation (25 episodes), and test (25 episodes) splits (check [PerAct's repo](https://github.com/peract/peract?tab=readme-ov-file#pre-generated-datasets) for details). The task list can be found in our paper.

For reference, I stored the dataset as:
```
[PERACT_DATASET_PATH]
└─ raw
  └─ train
    └─ [TASK_1]
    └─ [TASK_2]
    └─ ...
  └─ val
    └─ [TASK_1]
    └─ [TASK_2]
    └─ ...
  └─ test
    └─ [TASK_1]
    └─ [TASK_2]
    └─ ...
```



Download [RLBench's object mesh](https://drive.google.com/drive/folders/1bupiLa2akr2sytb7jcULnU_6ed4OOgkw?usp=sharing) or manually export the object mesh from CoppeliaSim.



### Dataset Generation

Set up the arguments in `scripts/gen_demonstration_rlbench.sh`.

-  `--peract_demo_dir` specifies the path to store PerAct's demo, e.g., `[PERACT_DATASET_PATH]/raw`.
-  `--save_path` specifies the path to store the generated dataset.

Run the script to collect demonstration from RLBench for all tasks.

```
bash scripts/gen_demonstration_rlbench.sh
```



### Policy Training

- Set `dataset.root_dir` in `config/task/rlbench_multi.yaml` to the path of generated demonstration, i.e., `--save_path` in the script `scripts/gen_demonstration_rlbench.sh`.
- (Optional) Modified `self.task_list` in `diffusion_policy_3d/dataset/rlbench_dataset_list.py` if you want to select your own task suite.
- (Optional) For single task training, set `dataset.root_dir` in `config/task/rlbench/[TASK_NAME].yaml` instead

Run the script for training:
```
# Train on all tasks
bash scripts/train_policy.sh rlbench_multi

# Train on single task
bash scripts/train_policy.sh rlbench/[TASK_NAME]
```



### Policy Evaluation

- Set `pose_estimation.mesh_dir` in `config/simple_dp3.yaml`, ensuring the path leads to the downloaded RLBench mesh file.
- (Optional) Modified task list in `scripts/eval_policy_multi.sh` if you want to select your own task suite.
- (Optional) Set `env_runner.root_dir` in `config/task/rlbench/[TASK_NAME].yaml` to the path of generated demonstration, i.e., `--save_path` in the script `scripts/gen_demonstration_rlbench.sh`.

Run the script for evaluation:
```
# Evaluate on all tasks
bash scripts/eval_policy_multi.sh

# Evaluate on single task
bash scripts/eval_policy.sh rlbench/[TASK_NAME]
```

- Note: The paper's results are based on an internal version of Foundation Pose that cannot be public released due to legal restrictions. Instead, we reference the public version of Foundation Pose. Our testing showed no performance degradation on the RLbench benchmark (See [here](https://github.com/NVlabs/FoundationPose?tab=readme-ov-file#notes) for more information).

---



## Training and Deployment on Real Robot

In this section, I describe my workflow for real-world experiments. This should serve only as a reference, and I recommend that readers use any tools they are familiar with. I used only one iPhone 12 Pro for the entire data collection process.



### Environment Setup

- This guide assumes that the conda environment `spot` has been configured according to the instructions in [Installation](https://github.com/NVlabs/object_centric_diffusion#Installation).
- (Optional) Set up another conda environment named `yolo_world` following [YOLO-World Installation](https://github.com/AILab-CVC/YOLO-World#1-installation).
  - YOLO-World is used to obtain object masks during training (see the script `env_real/data/prepare_mask.py`) and deployment. If you use a different object detection/segmentation model, you can ignore this step.

### Dataset Collection
For policy training and deployment, we need the following:
- **Object mesh** for pose tracking
- **Task demonstration video** for policy training

For each task, the object mesh is the reconstructed mesh of the graspable object (e.g., pitcher) and the target object (e.g., cup). The task demonstration is an RGBD video, where a human hand performs the task (e.g., pour water).

1. Object mesh scanning
    Use [AR Code](https://apps.apple.com/us/app/ar-code-object-capture-3d-scan) to scan both graspable and target objects. Export the mesh in `.usdz` format. Uncompress the `.usdz` file to obtain the .usdc mesh file, then convert the `.usdc` file to `.obj` format. I personally use Blender for this conversion process.
2. Task demonstration collection
    Use [Record3D](https://apps.apple.com/us/app/record3d-3d-videos) to shoot a demonstration video of a single human hand performing the task. Use the export option "EXR + JPG sequence" to get the `.r3d` file.


After completing steps 1 and 2, place the mesh file in the `mesh` directory and the r3d file in the `r3d` directory. The files should be stored as follows:

```
[TASK_DATASET_PATH]
└─ mesh
  └─ pitcher
    └─ pitcher.obj
  └─ cup
    └─ cup.obj
└─ r3d
  └─ 2024-09-07--01-11-23.r3d
  └─ 2024-09-07--01-11-41.r3d
  └─ ...
```



### Dataset Post-Processing

Set up the task name and object name using `real_task_object_dict` in `env_real/utils/realworld_objects.py`.
- The `grasp_object_name` and `target_object_name` should be consistent with the folder names under `[TASK_DATASET_PATH]/mesh`.
- The `grasp_object_prompt` and `target_object_prompt` are the prompts for the object detection/segmentation model (in this case, Yolo-World) to obtain the object bounding box/mask for tracking.

Run the script for dataset generation:
```
bash scripts/gen_demonstration_real.sh
```

The generated dataset will be saved under `[TASK_DATASET_PATH]/zarr`. 



### Policy Training

To train the policy, set `dataset.root_dir` to `[TASK_DATASET_PATH]` in the config file (see `config/task/_real_world_task_template.yaml` for details).

Run the script for training:

```
bash scripts/train_policy.sh [TASK_NAME]
```



## Troubleshooting
ModuleNotFoundError: No module named 'rlbench.action_modes'
- Edit "setup.py" RLBench library and add 'rlbench.action_modes'. See [here](https://github.com/stepjam/RLBench/issues/160) for more details.



## Acknowledge
The policy learning is based on [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy). The RLBench data collection and evaluation is based on [RLBench](https://github.com/stepjam/RLBench) and [PerAct](https://github.com/peract/peract). The object pose tracking is based on [FoundationPose](https://github.com/NVlabs/FoundationPose). Thanks for their wonderful work.

## License

The code and data are released under the NVIDIA Source Code License. Copyright © 2025, NVIDIA Corporation. All rights reserved.
