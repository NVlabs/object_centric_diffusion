DEBUG=False
seed=0

alg_name=simple_dp3
task_name=${1}
config_name=${alg_name}
seed=${seed}
exp_name=${task_name}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=0
use_fp=True # if false use ground-truth object pose instead
eval_epoch=1000


export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python tools/eval_dp3.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.env_runner.use_fp=${use_fp} \
                            evaluation.eval_epoch=${eval_epoch}
